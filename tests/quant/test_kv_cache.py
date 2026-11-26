"""Unit tests for KV-cache quantization simulation (design doc §6, §10.1
TestKVSim)."""

import pytest
import torch
from torch import nn

from xtuner.v1.quant.config import KVQConfig, QADQuantConfig
from xtuner.v1.quant.fake_quant_linear import apply_quantization
from xtuner.v1.quant.kv_cache import (
    KVCacheQuantizer,
    apply_kv_quantization,
    remove_kv_quantization,
)


EPS_FP32 = torch.finfo(torch.float32).eps


def fake_quant_reference(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Deploy-IR numerics without autograd: clamp(round_half_even(x/scale),
    [-128, 127]) * scale."""
    x_f32 = x.detach().float()
    q = torch.clamp(torch.round(x_f32 / scale), -128, 127)
    return (q * scale).to(x.dtype)


class _StubAttention(nn.Module):
    """Stub attention whose forward receives K and V directly, exercising arg
    interception."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, k: torch.Tensor, v: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return (k * v).sum(dim=-1, keepdim=True) * self.gate * alpha


def expected_stub_output(
    stub: _StubAttention, k: torch.Tensor, v: torch.Tensor, alpha: float, decay: float
) -> torch.Tensor:
    """Single-quantization reference computed with a fresh quantizer
    replicating the EMA state."""
    ref = KVCacheQuantizer(ema_decay=decay)
    return (ref.quantize_key(k) * ref.quantize_value(v)).sum(dim=-1, keepdim=True) * stub.gate.detach() * alpha


class _DummySelfAttention(nn.Module):
    """Plain QKV attention computing K/V internally from ``k_proj``/``v_proj``
    children.

    ``capture`` optionally records the K/V tensors right after their projections, so tests can
    bit-compare the pre- and post-attach values entering the attention computation.

    Args:
        hidden_size (int): Input hidden size.
        kv_heads (int): Number of K/V heads; query head count is fixed at 2.
        head_dim (int): Per-head width of the projection outputs.
        seq_len (int): Sequence length; the block operates on token-major ``(B*S, H)`` activations,
            the 2-D shape ``IntQuantizer`` per-token activation quantization is defined on.
    """

    def __init__(self, hidden_size: int = 8, kv_heads: int = 2, head_dim: int = 4, seq_len: int = 5) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_key_value_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, 2 * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_heads * head_dim, bias=False)
        self.capture: dict[str, torch.Tensor] | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        t = hidden.shape[0]
        b = t // self.seq_len
        s = self.seq_len
        q = self.q_proj(hidden).view(b, s, 2, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(b, s, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(b, s, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.capture is not None:
            self.capture["k"] = k.detach().clone()
            self.capture["v"] = v.detach().clone()
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)  # (B, 2, S, head_dim)
        return out.transpose(1, 2).reshape(b * s, 2 * self.head_dim)


class _DummyBlock(nn.Module):
    """Decoder block whose attention child is named ``self_attn``, reproducing
    the T7 module tree.

    After ``apply_quantization``, ``named_modules()`` surfaces quantizer-owned paths like
    ``"self_attn.k_proj._act_quantizer"`` that contain the ``self_attn`` segment.
    """

    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.self_attn = _DummySelfAttention(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=False),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(self.self_attn(hidden))


class TestAxisContract:
    def test_hand_computed_transpose_reshape_dequant(self) -> None:
        # (B=1, H_kv=2, S=2, D=2); channel (h, d) amaxes differ so a wrong reduction axis shows up.
        x = torch.tensor([[[[3.0, 0.4], [-1.5, -0.4]], [[1.25, 6.0], [-2.5, -6.0]]]])
        q = KVCacheQuantizer(ema_decay=0.999)
        out = q.quantize_key(x)
        scale = q.k_scale
        assert scale is not None
        assert scale.shape == (2 * 2,)  # (H_kv*D,)
        expected_scale = torch.tensor([3.0, 0.4, 2.5, 6.0]) / 127.0
        assert torch.allclose(scale, expected_scale, atol=1e-7)
        # Hand-written transpose+reshape result: (B, H, S, D) -> (B*S=2 rows, H*D=4 channels).
        flat = torch.tensor([[3.0, 0.4, 1.25, 6.0], [-1.5, -0.4, -2.5, -6.0]])
        expected_flat = fake_quant_reference(flat, expected_scale.unsqueeze(0))
        expected = expected_flat.view(1, 2, 2, 2).permute(0, 2, 1, 3)
        assert torch.allclose(out, expected, atol=1e-6)
        # Spot-check the hand-computed 63.5-level dequant in channel (h=1, d=0):
        # 1.25/(2.5/127) = 63.5 -> nearest even 64 -> 64*(2.5/127) ~= 1.2598425.
        assert out[0, 1, 0, 0].item() == pytest.approx(64 * 2.5 / 127, abs=1e-6)
        assert out[0, 0, 1, 0].item() == pytest.approx(-64 * 3.0 / 127, abs=1e-6)

    def test_scale_reduced_over_batch_and_seq(self) -> None:
        # Two batch steps with different per-step magnitudes; the scale must see both (dim=0).
        x = torch.empty(2, 2, 3, 2)
        x[0] = 1.0
        x[1] = -9.0
        q = KVCacheQuantizer(ema_decay=0.999)
        q.quantize_key(x)
        scale = q.k_scale
        assert scale is not None
        assert scale.shape == (4,)
        assert torch.allclose(scale, torch.full((4,), 9.0 / 127.0), atol=1e-7)

    def test_wrong_axis_gives_different_statistics(self) -> None:
        x = torch.tensor([[[[3.0, 0.4], [-1.5, -0.4]], [[1.25, 6.0], [-2.5, -6.0]]]])
        q = KVCacheQuantizer(ema_decay=0.999)
        q.quantize_key(x)
        scale = q.k_scale
        assert scale is not None
        # The naive no-transpose reshape (B*H, S*D) reduces over the wrong axis and must not
        # reproduce the per-(head, head_dim) contract.
        naive = x.reshape(2, 4)
        naive_amax = torch.maximum(naive.min(dim=1).values.clamp(max=0).neg(), naive.max(dim=1).values.clamp(min=0))
        assert naive_amax.shape != scale.shape or not torch.allclose(naive_amax / 127.0, scale)
        assert torch.allclose(naive_amax, torch.tensor([3.0, 6.0]), atol=1e-7)

    def test_eps_floor_and_zero_containment(self) -> None:
        q = KVCacheQuantizer(ema_decay=0.999)
        tiny = torch.zeros(1, 2, 3, 2)
        tiny[0, 0, 0, 0] = 1e-30
        q.quantize_key(tiny)
        scale = q.k_scale
        assert scale is not None
        assert torch.all(scale == EPS_FP32)
        negative_only = torch.full((1, 1, 2, 2), -5.0)
        q.quantize_value(negative_only)
        v_scale = q.v_scale
        assert v_scale is not None
        assert torch.allclose(v_scale, torch.full((2,), 5.0 / 127.0), atol=1e-7)

    def test_non_4d_input_rejected(self) -> None:
        q = KVCacheQuantizer(ema_decay=0.999)
        with pytest.raises(ValueError, match="4-D"):
            q.quantize_key(torch.randn(2, 3))
        with pytest.raises(ValueError, match="4-D"):
            q.quantize_value(torch.randn(1, 2, 3, 4, 5))


class TestIndependentKV:
    def test_v_precision_unaffected_by_k_amax(self) -> None:
        q = KVCacheQuantizer(ema_decay=0.999)
        k = torch.tensor([[[[100.0, -100.0]]]])  # large key amax
        v = torch.tensor([[[[0.3, -0.3]]]])  # small value amax
        k_out = q.quantize_key(k)
        v_out = q.quantize_value(v)
        assert q.k_scale is not None and torch.allclose(q.k_scale, torch.full((2,), 100.0 / 127.0), atol=1e-6)
        assert q.v_scale is not None and torch.allclose(q.v_scale, torch.full((2,), 0.3 / 127.0), atol=1e-6)
        # With its own fine scale, V round-trips; a shared K scale would quantize 0.3 to 0.
        assert torch.allclose(v_out, v, atol=1e-6)
        assert v_out.abs().sum() > 0
        assert torch.allclose(k_out, k, atol=1e-6)

    def test_scales_are_two_independent_sets(self) -> None:
        torch.manual_seed(7)
        q = KVCacheQuantizer(ema_decay=0.999)
        k = torch.randn(2, 3, 4, 2)
        v = torch.randn(2, 3, 4, 2) * 0.01
        q.quantize_key(k)
        q.quantize_value(v)
        assert q.k_scale is not None and q.v_scale is not None
        assert not torch.allclose(q.k_scale, q.v_scale)
        # Updating K must not move V and vice versa.
        k_before = q.k_scale.clone()
        v_before = q.v_scale.clone()
        q.quantize_key(torch.randn(2, 3, 4, 2) * 50)
        assert torch.equal(q.v_scale, v_before)
        assert not torch.equal(q.k_scale, k_before)
        q.quantize_value(torch.randn(2, 3, 4, 2))
        assert torch.equal(k_before, q.k_scale) or not torch.equal(q.v_scale, v_before)


class TestEMAAndFreeze:
    def test_ema_follows_formula(self) -> None:
        q = KVCacheQuantizer(ema_decay=0.9)
        q.quantize_key(torch.tensor([[[[4.0]]]]))
        assert torch.allclose(q.k_scale, torch.tensor(4.0 / 127.0), atol=1e-7)  # init ema = first amax
        q.quantize_key(torch.tensor([[[[8.0]]]]))
        assert torch.allclose(q.k_scale, torch.tensor((0.9 * 4.0 + 0.1 * 8.0) / 127.0), atol=1e-7)
        q.quantize_key(torch.tensor([[[[2.0]]]]))
        assert torch.allclose(q.k_scale, torch.tensor((0.9 * 4.4 + 0.1 * 2.0) / 127.0), atol=1e-7)

    def test_ema_independent_per_kv(self) -> None:
        q = KVCacheQuantizer(ema_decay=0.5)
        q.quantize_key(torch.tensor([[[[8.0]]]]))
        q.quantize_value(torch.tensor([[[[2.0]]]]))
        assert torch.allclose(q.k_scale, torch.tensor(8.0 / 127.0), atol=1e-7)
        assert torch.allclose(q.v_scale, torch.tensor(2.0 / 127.0), atol=1e-7)

    def test_frozen_scale_constant_across_new_inputs(self) -> None:
        q = KVCacheQuantizer(ema_decay=0.9)
        q.quantize_key(torch.tensor([[[[4.0]]]]))
        q.quantize_key(torch.tensor([[[[8.0]]]]))
        frozen_scale = q.k_scale.clone()
        q.set_frozen(True)
        assert q.frozen
        q.quantize_key(torch.tensor([[[[100.0]]]]))
        assert torch.equal(q.k_scale, frozen_scale)
        # Outputs stay on the frozen grid: 50/scale clamps to the top int8 level.
        out = q.quantize_key(torch.tensor([[[[50.0]]]]))
        assert torch.allclose(out, torch.tensor([[[[127.0 * frozen_scale[0]]]]]), atol=1e-6)
        # Unfreeze resumes EMA updates.
        q.set_frozen(False)
        assert not q.frozen
        q.quantize_key(torch.tensor([[[[8.0]]]]))
        assert torch.allclose(q.k_scale, torch.tensor((0.9 * 4.4 + 0.1 * 8.0) / 127.0), atol=1e-7)

    def test_frozen_before_calibration_rejected(self) -> None:
        q = KVCacheQuantizer(ema_decay=0.9)
        q.set_frozen(True)
        with pytest.raises(RuntimeError, match="frozen"):
            q.quantize_key(torch.tensor([[[[1.0]]]]))

    def test_no_state_dict_leak(self) -> None:
        q = KVCacheQuantizer(ema_decay=0.999)
        q.quantize_key(torch.randn(1, 2, 3, 4))
        q.quantize_value(torch.randn(1, 2, 3, 4))
        assert len(q.state_dict()) == 0
        assert len(list(q.parameters())) == 0


class TestGradient:
    def test_ste_backward_identity_in_range(self) -> None:
        q = KVCacheQuantizer(ema_decay=0.999)
        k = torch.randn(2, 2, 3, 4, requires_grad=True)
        out = q.quantize_key(k)
        out.sum().backward()
        assert k.grad is not None
        # Self-calibrated scale -> every |x| <= amax -> identity straight-through.
        assert torch.equal(k.grad, torch.ones_like(k))
        v = torch.randn(2, 2, 3, 4, requires_grad=True)
        q.quantize_value(v).sum().backward()
        assert v.grad is not None and torch.equal(v.grad, torch.ones_like(v))

    def test_ste_clip_mask_on_frozen_scale(self) -> None:
        q = KVCacheQuantizer(ema_decay=0.999)
        q.quantize_key(torch.ones(1, 1, 1, 2))  # scale = 1/127
        q.set_frozen(True)
        x = torch.tensor([[[[0.5, 5.0]]]], requires_grad=True)  # 5.0 exceeds 127*(1/127)
        out = q.quantize_key(x)
        out.sum().backward()
        assert torch.equal(x.grad, torch.tensor([[[[1.0, 0.0]]]]))

    def test_gradient_flows_through_k_proj_style_linear(self) -> None:
        torch.manual_seed(3)
        k_proj = nn.Linear(8, 4, bias=False)
        x = torch.randn(2, 5, 8, requires_grad=True)
        k = k_proj(x).view(2, 5, 2, 2).transpose(1, 2)
        q = KVCacheQuantizer(ema_decay=0.999)
        out = q.quantize_key(k)
        ref = (
            fake_quant_reference(k.detach().permute(0, 2, 1, 3).reshape(10, 4), q.k_scale.unsqueeze(0))
            .view(2, 5, 2, 2)
            .permute(0, 2, 1, 3)
        )
        assert torch.allclose(out.detach(), ref, atol=1e-6)
        out.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0
        assert k_proj.weight.grad is not None and k_proj.weight.grad.abs().sum() > 0


class TestAttachDetach:
    def test_stub_forward_outputs_quantized(self) -> None:
        torch.manual_seed(11)
        stub = _StubAttention()
        handles = apply_kv_quantization(stub, KVQConfig(), kv_amax_ema_decay=0.5)
        assert len(handles) == 1 and handles[0].module is stub
        k = torch.randn(2, 2, 3, 4)
        v = torch.randn(2, 2, 3, 4)
        # Same call via positional args, keyword args, and a non-tensor extra positional.
        for alpha in (1.0, 2.0):
            assert torch.equal(stub(k, v, alpha), expected_stub_output(stub, k, v, alpha, 0.5))
            assert torch.equal(stub(k=k, v=v, alpha=alpha), expected_stub_output(stub, k, v, alpha, 0.5))

    def test_stub_quantization_uses_kv_entry_before_compute(self) -> None:
        torch.manual_seed(12)
        stub = _StubAttention()
        apply_kv_quantization(stub, KVQConfig())
        k = torch.randn(1, 1, 4, 4)
        v = torch.randn(1, 1, 4, 4)
        out = stub(k, v)
        expected = expected_stub_output(stub, k, v, 1.0, 0.999)
        assert torch.equal(out, expected)
        raw = (k * v).sum(dim=-1, keepdim=True) * stub.gate.detach()
        assert not torch.equal(out, raw)

    def test_detach_restores_bit_identical_outputs(self) -> None:
        torch.manual_seed(13)
        stub = _StubAttention()
        k = torch.randn(2, 2, 3, 4)
        v = torch.randn(2, 2, 3, 4)
        original = stub(k, v)
        handles = apply_kv_quantization(stub, KVQConfig())
        quantized = stub(k, v)
        assert not torch.equal(quantized, original)
        handles[0].detach()
        restored = stub(k, v)
        assert torch.equal(restored, original)
        # detach is idempotent
        handles[0].detach()
        assert torch.equal(stub(k, v), original)

    def test_attach_is_idempotent(self) -> None:
        torch.manual_seed(14)
        stub = _StubAttention()
        first = apply_kv_quantization(stub, KVQConfig(), kv_amax_ema_decay=0.7)
        second = apply_kv_quantization(stub, KVQConfig(), kv_amax_ema_decay=0.7)
        assert second == first
        k = torch.randn(2, 2, 3, 4)
        v = torch.randn(2, 2, 3, 4)
        out = stub(k, v)
        # A second wrap would quantize twice and deviate; a single wrap matches the reference.
        assert torch.equal(out, expected_stub_output(stub, k, v, 1.0, 0.7))

    def test_remove_kv_quantization_restores_model(self) -> None:
        torch.manual_seed(15)
        model = nn.Sequential(_StubAttention(), _StubAttention())
        handles = apply_kv_quantization(model, KVQConfig())
        assert len(handles) == 2
        k = torch.randn(1, 1, 2, 2)
        v = torch.randn(1, 1, 2, 2)
        quantized = model[0](k, v)
        detached = remove_kv_quantization(model)
        assert len(detached) == 2
        assert torch.equal(model[0](k, v), (k * v).sum(dim=-1, keepdim=True) * model[0].gate.detach())
        assert not torch.equal(quantized, model[0](k, v))
        assert remove_kv_quantization(model) == []

    def test_no_state_dict_leak_after_attach(self) -> None:
        stub = _StubAttention()
        handles = apply_kv_quantization(stub, KVQConfig())
        stub(torch.randn(1, 1, 2, 2), torch.randn(1, 1, 2, 2))
        assert set(stub.state_dict()) == {"gate"}
        assert len(handles[0].quantizer.state_dict()) == 0
        assert len(list(handles[0].quantizer.parameters())) == 0

    def test_unmatched_module_raises(self) -> None:
        class WeirdAttention(nn.Module):
            def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # no k/v args, no projections
                return hidden

        model = nn.Sequential(WeirdAttention())
        with pytest.raises(ValueError, match="Weird"):
            apply_kv_quantization(model, KVQConfig())


class TestAttachAfterWeightActQuant:
    """KV attach must keep selecting only real attention roots once
    ``apply_quantization`` has registered quantizer-owned submodules under the
    ``self_attn`` name path (T7 misfire)."""

    def test_kv_attach_skips_quantizer_submodules_and_hits_real_attention(self) -> None:
        torch.manual_seed(31)
        model = _DummyBlock()
        x = torch.randn(10, model.self_attn.q_proj.in_features)  # token-major (B*S, H)

        # Trap setup: weight/act quantization first, so IntQuantizer submodules appear under
        # "...self_attn.k_proj._act_quantizer" / "..._weight_quantizer" paths.
        apply_quantization(model, QADQuantConfig.for_schema("w8a8"))
        quantizer_paths = [name for name, _ in model.named_modules() if "_quantizer" in name]
        assert any("self_attn" in name for name in quantizer_paths)  # the misfire candidates exist

        capture_pre: dict[str, torch.Tensor] = {}
        model.self_attn.capture = capture_pre
        out_pre = model(x)
        assert {"k", "v"} <= capture_pre.keys()

        handles = apply_kv_quantization(model, KVQConfig(), kv_amax_ema_decay=0.5)
        # Exactly one handle, bound to the real attention ROOT module, never a quantizer-owned path.
        assert len(handles) == 1
        assert handles[0].module is model.self_attn
        name_by_id = {id(m): n for n, m in model.named_modules()}
        handle_paths = [name_by_id[id(h.module)] for h in handles]
        assert all("_quantizer" not in p for p in handle_paths)

        # K/V interception lands on the real attention forward: the captured K/V are quantized
        # bit-exactly per the KV sim, while the pre-attach run left them untouched.
        capture_post: dict[str, torch.Tensor] = {}
        model.self_attn.capture = capture_post
        out_post = model(x)
        assert not torch.equal(capture_post["k"], capture_pre["k"])
        assert not torch.equal(capture_post["v"], capture_pre["v"])
        ref = KVCacheQuantizer(ema_decay=0.5)
        assert torch.equal(capture_post["k"], ref.quantize_key(capture_pre["k"]))
        assert torch.equal(capture_post["v"], ref.quantize_value(capture_pre["v"]))
        assert not torch.equal(out_post, out_pre)
        assert handles[0].quantizer.k_scale is not None and handles[0].quantizer.v_scale is not None

        # Idempotent re-attach: the stored "_qad_kv_quantizer" submodule must not be re-selected.
        assert apply_kv_quantization(model, KVQConfig(), kv_amax_ema_decay=0.5) == handles
        model.self_attn.capture = {}
        assert torch.equal(model(x), out_post)

        # Detach restores the (still weight/act-quantized) model bit-exactly.
        assert remove_kv_quantization(model) == handles
        model.self_attn.capture = {}
        assert torch.equal(model(x), out_pre)
        assert remove_kv_quantization(model) == []


class TestRealAttention:
    """Forward-compat wiring checks against real xtuner attention classes.

    The real classes compute K/V internally (no forward k/v arguments), so quantization must be wired at the projection
    boundary. CPU boxes lack flash-attn/lmdeploy kernels; the tests stand in the kernel symbols with pure-torch
    captures so the real forwards run end to end.
    """

    def test_mla_forward_quantizes_kv_and_keeps_q(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mla_mod = pytest.importorskip("xtuner.v1.module.attention.mla")
        SequenceContext = mla_mod.SequenceContext
        attn = mla_mod.MLAConfig(
            num_attention_heads=2,
            head_dim=16,
            kv_lora_rank=8,
            q_lora_rank=None,
            qk_rope_head_dim=2,
            qk_nope_head_dim=6,
            v_head_dim=4,
        ).build(hidden_size=8)
        handles = apply_kv_quantization(attn, KVQConfig())
        assert len(handles) == 1

        s, hidden = 5, 8
        captured: dict[str, torch.Tensor] = {}

        def fake_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> torch.Tensor:
            captured["q"], captured["k"], captured["v"] = q.detach().clone(), k.detach().clone(), v.detach().clone()
            scores = torch.einsum("shd,thd->hst", q, k) * kwargs["softmax_scale"]
            mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            return torch.einsum("hst,thd->shd", probs, v)

        monkeypatch.setattr(mla_mod, "flash_attn_varlen_func", fake_flash)
        torch.manual_seed(21)
        x = torch.randn(1, s, hidden)
        cos = torch.randn(1, s, 2)
        sin = torch.randn(1, s, 2)
        cu = torch.tensor([0, s], dtype=torch.int32)
        seq_ctx = SequenceContext(
            input_ids=torch.zeros(1, s, dtype=torch.long),
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=s,
            max_length_k=s,
            device="cpu",
        )
        out = attn.forward_training(x, (cos, sin), seq_ctx)
        assert torch.isfinite(out).all()

        # Independent scale shapes: (H_kv * qk_nope,) for K and (H_kv * v_head,) for V.
        quantizer = handles[0].quantizer
        assert quantizer.k_scale is not None and quantizer.k_scale.shape == (2 * 6,)
        assert quantizer.v_scale is not None and quantizer.v_scale.shape == (2 * 4,)

        # Manual replication of the real forward up to the kernel boundary.
        compressed_kv = attn.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(compressed_kv, [8, 2], dim=-1)
        k_pe = k_pe.view(1, s, 1, 2).transpose(1, 2)
        kv_raw = torch.nn.functional.linear(attn.kv_a_layernorm(compressed_kv), attn.kv_b_proj.weight)
        kv_raw = kv_raw.view(1, s, 2, 10).transpose(1, 2)
        k_nope_raw, v_raw = torch.split(kv_raw, [6, 4], dim=-1)
        ref = KVCacheQuantizer(ema_decay=0.999)
        k_nope_q = ref.quantize_key(k_nope_raw)
        v_q = ref.quantize_value(v_raw)
        q_raw = attn.q_proj(x).view(1, s, 2, 8).transpose(1, 2)
        q_nope, q_pe = torch.split(q_raw, [6, 2], dim=-1)
        q_pe_r, k_pe_r = mla_mod.mla_apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        k_manual = torch.empty(1, 2, s, 8)
        k_manual[..., :6], k_manual[..., 6:] = k_nope_q, k_pe_r
        v_manual = torch.nn.functional.pad(v_q, [0, 8 - 4])
        q_manual = torch.empty(1, 2, s, 8)
        q_manual[..., :6], q_manual[..., 6:] = q_nope, q_pe_r

        assert torch.allclose(captured["q"], q_manual.transpose(1, 2).squeeze(0), atol=1e-6)  # Q untouched
        assert torch.allclose(captured["k"], k_manual.transpose(1, 2).squeeze(0), atol=1e-6)
        assert torch.allclose(captured["v"], v_manual.transpose(1, 2).squeeze(0), atol=1e-6)
        # K/V dequantized values sit on the deploy int8 grid of their independent scales.
        ratio_v = captured["v"][..., :4] / quantizer.v_scale.view(1, 2, 4)
        assert torch.allclose(ratio_v, ratio_v.round(), atol=1e-4)
        ratio_k = captured["k"][..., :6] / quantizer.k_scale.view(1, 2, 6)
        assert torch.allclose(ratio_k, ratio_k.round(), atol=1e-4)

    def test_mha_forward_quantizes_kv_at_projections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mha_mod = pytest.importorskip("xtuner.v1.module.attention.mha")
        SequenceContext = mha_mod.SequenceContext
        attn = mha_mod.MHAConfig(
            num_attention_heads=2, num_key_value_heads=2, head_dim=4, attn_impl="eager_attention"
        ).build(hidden_size=8)
        handles = apply_kv_quantization(attn, KVQConfig())
        assert len(handles) == 1
        assert handles[0].quantizer.k_scale is None  # nothing calibrated before the first forward

        s = 5
        captured: dict[str, torch.Tensor] = {}

        def fake_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> torch.Tensor:
            captured["q"], captured["k"], captured["v"] = q.detach().clone(), k.detach().clone(), v.detach().clone()
            scores = torch.einsum("shd,thd->hst", q, k) * attn.scaling
            mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            return torch.einsum("hst,thd->shd", probs, v)

        monkeypatch.setattr(mha_mod, "fill_paged_kv_cache", lambda *args: None)
        monkeypatch.setattr(mha_mod, "flash_attn_varlen_func", fake_flash)
        torch.manual_seed(22)
        x = torch.randn(1, s, 8)
        cos = torch.randn(1, s, 4)
        sin = torch.randn(1, s, 4)
        cu = torch.tensor([0, s], dtype=torch.int32)
        seq_ctx = SequenceContext(
            input_ids=torch.zeros(1, s, dtype=torch.long),
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=s,
            max_length_k=s,
            device="cpu",
        )
        past_key_values = [[torch.zeros(1, s, 2, 4), torch.zeros(1, s, 2, 4)]]
        out = attn.prefilling(x, (cos, sin), seq_ctx, past_key_values)
        assert torch.isfinite(out).all()

        quantizer = handles[0].quantizer
        assert quantizer.k_scale is not None and quantizer.k_scale.shape == (2 * 4,)
        assert quantizer.v_scale is not None and quantizer.v_scale.shape == (2 * 4,)

        # Manual replication: k_proj/v_proj outputs quantized pre-rope, q untouched.
        k_raw = torch.nn.functional.linear(x, attn.k_proj.weight).view(1, s, 2, 4).transpose(1, 2)
        v_raw = torch.nn.functional.linear(x, attn.v_proj.weight).view(1, s, 2, 4).transpose(1, 2)
        q_raw = attn.q_proj(x).view(1, s, 2, 4).transpose(1, 2)
        ref = KVCacheQuantizer(ema_decay=0.999)
        k_q = ref.quantize_key(k_raw)
        v_q = ref.quantize_value(v_raw)
        q_r, k_r = attn.apply_rotary_emb(q_raw, k_q, cos, sin)
        assert torch.allclose(captured["q"], q_r.transpose(1, 2).squeeze(0), atol=1e-6)
        assert torch.allclose(captured["k"], k_r.transpose(1, 2).squeeze(0), atol=1e-6)
        assert torch.allclose(captured["v"], v_q.transpose(1, 2).squeeze(0), atol=1e-6)
        ratio = captured["v"] / quantizer.v_scale.view(1, 2, 4)
        assert torch.allclose(ratio, ratio.round(), atol=1e-4)

    def test_real_class_detach_restores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mha_mod = pytest.importorskip("xtuner.v1.module.attention.mha")
        attn = mha_mod.MHAConfig(
            num_attention_heads=2, num_key_value_heads=2, head_dim=4, attn_impl="eager_attention"
        ).build(hidden_size=8)
        x = torch.randn(2, 3, 8)
        original_k = attn.k_proj(x)
        handles = apply_kv_quantization(attn, KVQConfig())
        quantized_k = attn.k_proj(x)
        assert not torch.equal(original_k, quantized_k)
        handles[0].detach()
        assert torch.equal(attn.k_proj(x), original_k)
        assert remove_kv_quantization(attn) == []
