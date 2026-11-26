"""Unit tests for FakeQuantLinear (design doc §5.2, §5.3, §10.1
TestFakeQuantLinear)."""

import pytest
import torch
from torch import nn

from xtuner.v1.quant.config import QADQuantConfig
from xtuner.v1.quant.fake_quant_linear import (
    FakeQuantLinear,
    apply_quantization,
    build_quantizers,
    remove_quantization,
)
from xtuner.v1.quant.quantizer import IntQuantizer


def fake_quant(x: torch.Tensor, quantizer: IntQuantizer) -> torch.Tensor:
    """Hand-rolled deploy-IR fake quant: clamp(round(x/scale)) * scale in x's
    dtype."""
    scale = quantizer.compute_scale(x)
    return quantizer.quant_dequant(x, scale)


class StubLoRA:
    """Stub adapter exposing ``delta_w()`` (B @ A); no training logic."""

    def __init__(self, lora_a: torch.Tensor, lora_b: torch.Tensor) -> None:
        self.lora_a = lora_a
        self.lora_b = lora_b

    def delta_w(self) -> torch.Tensor:
        return self.lora_b @ self.lora_a


class TestFakeQuantLinear:
    def test_forward_matches_hand_computed(self) -> None:
        torch.manual_seed(0)
        cfg = QADQuantConfig(quant_schema="w8a8")
        lin = FakeQuantLinear(8, 4, quant_cfg=cfg)
        x = torch.randn(3, 8)
        w_q = fake_quant(lin.weight, IntQuantizer(8, "per_channel"))
        x_q = fake_quant(x, IntQuantizer(8, "per_token"))
        expected = torch.nn.functional.linear(x_q, w_q, lin.bias)
        assert torch.equal(lin(x), expected)

    def test_w4a8_quantizer_selection(self) -> None:
        wq, aq = build_quantizers(QADQuantConfig(quant_schema="w4a8", w4_granularity="per_group", group_size=4))
        assert (wq.dtype, wq.scope, wq.group_size) == ("int4", "per_group", 4)  # type: ignore[attr-defined]
        assert (aq.dtype, aq.scope) == ("int8", "per_batch")  # type: ignore[attr-defined]
        wq2, aq2 = build_quantizers(QADQuantConfig(quant_schema="w4a8", w4_granularity="per_channel"))
        assert (wq2.dtype, wq2.scope) == ("int4", "per_channel")  # type: ignore[attr-defined]
        assert (aq2.dtype, aq2.scope) == ("int8", "per_batch")  # type: ignore[attr-defined]

    def test_scale_cache_reused_within_step(self) -> None:
        torch.manual_seed(0)
        lin = FakeQuantLinear(8, 4, quant_cfg=QADQuantConfig(quant_schema="w8a8"))
        x = torch.randn(3, 8)
        lin.begin_step()
        out1 = lin(x)
        assert lin._w_scale_cache is not None
        cached_scale = lin._w_scale_cache.clone()
        with torch.no_grad():
            lin.weight.copy_(torch.randn_like(lin.weight) * 3.0)
        # no begin_step: cached scale reused (identical scale tensor; output changes because the
        # new weight is quantized with the stale scale)
        assert torch.equal(lin._w_scale_cache, cached_scale)
        assert not torch.equal(lin(x), out1)
        # begin_step: scale recomputed from the new weight -> different scale
        lin.begin_step()
        assert lin(x) is not None
        assert not torch.equal(lin._w_scale_cache, cached_scale)

    def test_freeze_scales_pins_scale_while_training(self) -> None:
        torch.manual_seed(1)
        lin = FakeQuantLinear(8, 4, quant_cfg=QADQuantConfig(quant_schema="w8a8"))
        x = torch.randn(3, 8)
        lin.freeze_scales()
        out1 = lin(x)
        assert lin._w_scale_cache is not None
        pinned = lin._w_scale_cache.clone()
        # (a) pinned scale unchanged under weight mutation (begin_step is a no-op while frozen).
        # Mutate within the pinned amax range and off the quantization grid so both the
        # output and the STE clip mask stay informative.
        with torch.no_grad():
            lin.weight.copy_(lin.weight * 0.5 + 0.25 * pinned.reshape(-1, 1).to(lin.weight.dtype))
        lin.begin_step()
        assert torch.equal(lin._w_scale_cache, pinned)
        # (b) output changes: current weight quantized with the fixed scale
        out2 = lin(x)
        assert not torch.equal(out2, out1)
        # (c) gradient still flows to the weight during freeze
        lin(x).sum().backward()
        assert lin.weight.grad is not None and torch.any(lin.weight.grad != 0)
        # (d) unfreeze resumes scale recomputation
        lin.unfreeze_scales()
        lin.begin_step()
        out3 = lin(x)
        assert lin._w_scale_cache is not None
        assert not torch.equal(lin._w_scale_cache, pinned)
        # scale actually changed -> recomputation resumed (output determinism keeps lin(x) == out3)
        assert torch.equal(lin(x), out3)

    def test_lora_merge_then_quant(self) -> None:
        torch.manual_seed(2)
        lin = FakeQuantLinear(8, 4, quant_cfg=QADQuantConfig(quant_schema="w8a8"))
        adapter = StubLoRA(torch.randn(2, 8), torch.randn(4, 2))
        lin.attach_lora(adapter)
        x = torch.randn(3, 8)
        w_eff = (lin.weight + adapter.delta_w()).detach()
        wq = IntQuantizer(8, "per_channel")
        xq = IntQuantizer(8, "per_token")
        expected = torch.nn.functional.linear(fake_quant(x, xq), fake_quant(w_eff, wq), lin.bias)
        assert torch.equal(lin(x), expected)

    def test_lora_not_output_add(self) -> None:
        # merge-then-quant must differ from quantizing W and adding quantized BA to the output
        torch.manual_seed(3)
        lin = FakeQuantLinear(8, 4, quant_cfg=QADQuantConfig(quant_schema="w8a8"))
        adapter = StubLoRA(torch.randn(2, 8), torch.randn(4, 2))
        lin.attach_lora(adapter)
        x = torch.randn(3, 8)
        wq = IntQuantizer(8, "per_channel")
        xq = IntQuantizer(8, "per_token")
        out = lin(x)
        output_add = torch.nn.functional.linear(fake_quant(x, xq), fake_quant(lin.weight.detach(), wq), lin.bias)
        output_add = output_add + x @ adapter.delta_w().detach().t()
        assert not torch.equal(out, output_add)

    def test_forward_3d_activation_shape_preserved(self) -> None:
        # Deploy-IR semantics: activations are flatten-quantized over (-1, H) rows and reshaped;
        # FakeQuantLinear must pass 3-D (B, S, H) inputs through with shape preserved.
        torch.manual_seed(6)
        lin = FakeQuantLinear(16, 16, quant_cfg=QADQuantConfig(quant_schema="w8a8"))
        x = torch.randn(2, 7, 16)
        out = lin(x)
        assert out.shape == (2, 7, 16)
        assert torch.isfinite(out).all()
        wq = IntQuantizer(8, "per_channel")
        xq = IntQuantizer(8, "per_token")
        x_q = fake_quant(x.reshape(-1, 16), xq).reshape(2, 7, 16)
        expected = torch.nn.functional.linear(x_q, fake_quant(lin.weight.detach(), wq), lin.bias)
        assert torch.equal(out, expected)

    def test_backward_3d_input_gradient_flows(self) -> None:
        lin = FakeQuantLinear(16, 8, quant_cfg=QADQuantConfig(quant_schema="w8a8"))
        x = torch.randn(2, 7, 16)
        lin(x).sum().backward()
        assert lin.weight.grad is not None and torch.any(lin.weight.grad != 0)

    def test_ste_gradient_flows_to_weight(self) -> None:
        lin = FakeQuantLinear(8, 4, quant_cfg=QADQuantConfig(quant_schema="w8a8"))
        x = torch.randn(3, 8, requires_grad=True)
        lin(x).sum().backward()
        assert lin.weight.grad is not None and torch.any(lin.weight.grad != 0)
        assert x.grad is not None and torch.any(x.grad != 0)

    def test_state_dict_keys_identical_and_loadable(self) -> None:
        torch.manual_seed(4)
        plain = nn.Linear(8, 4)
        quant_cfg = QADQuantConfig(quant_schema="w4a8", w4_granularity="per_group", group_size=4)
        apply_quantization(plain, quant_cfg)
        assert set(plain.state_dict().keys()) == {"weight", "bias"}
        # plain -> quantized loadability
        plain2 = nn.Linear(8, 4)
        plain2.load_state_dict(plain.state_dict())
        # quantized -> plain loadability
        lin = nn.Linear(8, 4)
        lin.load_state_dict(plain.state_dict())
        assert isinstance(lin, nn.Linear) and not isinstance(lin, FakeQuantLinear)

    def test_apply_remove_roundtrip(self) -> None:
        torch.manual_seed(5)
        lin = nn.Linear(8, 4)
        apply_quantization(lin, QADQuantConfig(quant_schema="w8a8"))
        assert isinstance(lin, FakeQuantLinear)
        remove_quantization(lin)
        assert type(lin) is nn.Linear
        assert not any(isinstance(m, FakeQuantLinear) for m in lin.modules())
        assert torch.equal(lin(torch.zeros(3, 8)), lin.bias.expand(3, 4))

    def test_apply_twice_raises(self) -> None:
        lin = nn.Linear(8, 4)
        apply_quantization(lin, QADQuantConfig(quant_schema="w8a8"))
        with pytest.raises(RuntimeError, match="already quantized"):
            apply_quantization(lin, QADQuantConfig(quant_schema="w8a8"))
