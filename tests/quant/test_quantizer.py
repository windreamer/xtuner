"""Unit tests for IntQuantizer (design doc §4.3, §5.1, §10.1
TestIntQuantizer)."""

import pytest
import torch

from xtuner.v1.quant import IntQuantizer


EPS_FP32 = torch.finfo(torch.float32).eps


def fake_quant_reference(x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """Deploy-IR numerics without autograd: clamp(round_half_even(x/scale),
    [qmin, qmax]) * scale."""
    x_f32 = x.detach().float()
    q = torch.clamp(torch.round(x_f32 / scale), qmin, qmax)
    return (q * scale).to(x.dtype)


class TestFormula:
    def test_zero_containment_all_negative(self) -> None:
        q = IntQuantizer(8, "per_channel")
        x = torch.tensor([[-5.0, -2.0, -1.0]])
        scale = q.compute_scale(x)
        assert torch.allclose(scale, torch.tensor([5.0 / 127.0]))

    def test_zero_containment_mixed_sign(self) -> None:
        q = IntQuantizer(8, "per_channel")
        x = torch.tensor([[-1.0, 3.0]])
        scale = q.compute_scale(x)
        assert torch.allclose(scale, torch.tensor([3.0 / 127.0]))

    def test_eps_floor(self) -> None:
        q = IntQuantizer(8, "per_channel")
        x = torch.tensor([[1e-30, -2e-31]])
        scale = q.compute_scale(x)
        assert scale.item() == EPS_FP32
        out = q.quant_dequant(x, scale)
        # tiny x / eps rounds to 0 -> output is 0
        assert torch.equal(out, torch.zeros_like(x))

    def test_eps_floor_clamps_to_bounds_int8(self) -> None:
        # frozen eps scale with large values clamps to the +-127 levels
        q = IntQuantizer(8, "per_channel")
        x = torch.tensor([[1.0, -1.0]])
        out = q.quant_dequant(x, torch.full_like(x, EPS_FP32))
        ref = fake_quant_reference(x, torch.full_like(x, EPS_FP32), -128, 127)
        assert torch.equal(out, ref)
        assert out[0, 0].item() == pytest.approx(127 * EPS_FP32, rel=1e-6)
        assert out[0, 1].item() == pytest.approx(-128 * EPS_FP32, rel=1e-6)

    def test_eps_floor_clamps_to_bounds_int4(self) -> None:
        q = IntQuantizer(4, "per_channel")
        x = torch.tensor([[1.0, -1.0]])
        out = q.quant_dequant(x, torch.full_like(x, EPS_FP32))
        ref = fake_quant_reference(x, torch.full_like(x, EPS_FP32), -8, 7)
        assert torch.equal(out, ref)
        assert out[0, 0].item() == pytest.approx(7 * EPS_FP32, rel=1e-6)
        assert out[0, 1].item() == pytest.approx(-8 * EPS_FP32, rel=1e-6)

    def test_round_ties_to_even_int8(self) -> None:
        # amax = 63.5 -> scale = 0.5; x/scale hits exact .5 boundaries
        q = IntQuantizer(8, "per_channel")
        x = torch.tensor([[63.5, 0.25, 0.75, -0.25, -0.75]])
        scale = q.compute_scale(x)
        assert scale.item() == 0.5
        out = q.quant_dequant(x, scale)
        # round-half-even: 127->127, 0.5->0, 1.5->2, -0.5->-0, -1.5->-2
        assert torch.equal(out, torch.tensor([[63.5, 0.0, 1.0, 0.0, -1.0]]))

    def test_round_ties_to_even_int4(self) -> None:
        # amax = 3.5 -> scale = 0.5
        q = IntQuantizer(4, "per_channel")
        x = torch.tensor([[3.5, 0.25, 0.75]])
        out = q.quant_dequant(x, q.compute_scale(x))
        assert torch.equal(out, torch.tensor([[3.5, 0.0, 1.0]]))

    def test_int8_quant_level_bounds(self) -> None:
        q = IntQuantizer(8, "per_channel")
        x = torch.tensor([[127.0, -127.0]])
        scale = q.compute_scale(x)
        assert scale.item() == 1.0
        out = q.quant_dequant(x, scale)
        assert torch.equal(out, torch.tensor([[127.0, -127.0]]))
        # out-of-range inputs under this frozen scale clamp to +-127 levels
        big = torch.tensor([[500.0, -500.0]])
        out_big = q.quant_dequant(big, scale)
        assert torch.equal(out_big, torch.tensor([[127.0, -128.0]]))

    def test_int4_quant_level_bounds(self) -> None:
        q = IntQuantizer(4, "per_channel")
        x = torch.tensor([[7.0, -7.0]])
        out = q.quant_dequant(x, q.compute_scale(x))
        assert torch.allclose(out, torch.tensor([[7.0, -7.0]]), atol=1e-5)


class TestScopes:
    def test_per_token_reduces_hidden_dim(self) -> None:
        q = IntQuantizer(8, "per_token")
        x = torch.tensor([[10.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        scale = q.compute_scale(x)
        assert scale.shape == (2, 1)
        assert torch.allclose(scale, torch.tensor([[10.0 / 127.0], [1.0 / 127.0]]))
        out = q.quant_dequant(x, scale)
        assert out.shape == x.shape

    def test_per_batch_reduces_batch_dim_w4a8_quirk(self) -> None:
        # W4A8 deployment quirk (§4.3.3): activation stats aggregate along dim=0 (batch).
        q = IntQuantizer(4, "per_batch")
        x = torch.tensor([[10.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        scale = q.compute_scale(x)
        assert scale.shape == (3,)
        assert torch.allclose(scale, torch.tensor([10.0 / 7.0, 1.0 / 7.0, EPS_FP32]))

    def test_per_token_vs_per_batch_differ(self) -> None:
        x = torch.tensor([[10.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        s_tok = IntQuantizer(8, "per_token").compute_scale(x)
        s_batch = IntQuantizer(8, "per_batch").compute_scale(x)
        assert s_tok.shape == (2, 1)
        assert s_batch.shape == (3,)
        assert torch.allclose(s_tok, torch.tensor([[10.0 / 127.0], [1.0 / 127.0]]))
        assert torch.allclose(s_batch, torch.tensor([10.0 / 127.0, 1.0 / 127.0, EPS_FP32]))

    def test_per_batch_3d_input(self) -> None:
        q = IntQuantizer(8, "per_batch")
        x = torch.ones(2, 3, 4) * torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = q.compute_scale(x)
        assert scale.shape == (4,)
        assert torch.allclose(scale, torch.tensor([1.0, 2.0, 3.0, 4.0]) / 127.0)
        out = q.quant_dequant(x, scale)
        assert out.shape == x.shape

    def test_per_token_3d_shape_preserved(self) -> None:
        # IR semantics (§4.3.2): quantize the (-1, H) view, reshape back; (B, S, H) must work.
        q = IntQuantizer(8, "per_token")
        torch.manual_seed(1)
        x = torch.randn(2, 5, 8)
        scale = q.compute_scale(x)
        assert scale.shape == (10, 1)
        out = q.quant_dequant(x, scale)
        assert out.shape == x.shape
        ref = fake_quant_reference(x.reshape(-1, 8), scale, -128, 127).reshape(x.shape)
        assert torch.equal(out, ref)

    def test_per_token_3d_backward(self) -> None:
        q = IntQuantizer(8, "per_token")
        x = torch.randn(2, 5, 8, requires_grad=True)
        out = q.quant_dequant(x, q.compute_scale(x))
        grad = torch.autograd.grad(out.sum(), x)[0]
        assert grad.shape == x.shape
        assert torch.allclose(grad, torch.ones_like(grad))

    def test_per_batch_3d_reduces_over_batch_times_seq(self) -> None:
        # per_batch must aggregate over B*S rows, not per (B, S) position: position (0, 1)
        # has the largest value 8, which only shows up when reducing over B*S.
        q = IntQuantizer(8, "per_batch")
        x = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0], [0.0, 8.0, 0.0]],
            ]
        )
        scale = q.compute_scale(x)
        assert scale.shape == (3,)
        assert torch.allclose(scale, torch.tensor([1.0 / 127.0, 8.0 / 127.0, EPS_FP32]))
        out = q.quant_dequant(x, scale)
        assert out.shape == x.shape
        ref = fake_quant_reference(x.reshape(-1, 3), scale, -128, 127).reshape(x.shape)
        assert torch.equal(out, ref)

    def test_per_channel_weight_scale_shape(self) -> None:
        q = IntQuantizer(8, "per_channel")
        w = torch.randn(6, 11)
        scale = q.compute_scale(w)
        assert scale.shape == (6,)
        amax_ref = w.abs().amax(dim=1)
        assert torch.allclose(scale, amax_ref / 127.0)

    def test_per_group_matches_reshape_semantics(self) -> None:
        # Groups are contiguous input-channel chunks within each output channel; with
        # in_features % group_size == 0 this equals the row-major partition
        # weight.reshape(-1, group_size) (== deploy-side weight.T.reshape(-1, group_size)
        # element partition, design §4.3.5).
        q = IntQuantizer(4, "per_group", group_size=4)
        w = torch.randn(3, 8)
        scale = q.compute_scale(w)
        assert scale.shape == (3, 2)
        grouped = w.reshape(-1, 4)
        amax_ref = torch.maximum(grouped.min(dim=1).values.clamp(max=0).neg(), grouped.max(dim=1).values.clamp(min=0))
        assert torch.allclose(scale.flatten(), amax_ref / 7.0)
        # dequant matches the reference formula per group
        out = q.quant_dequant(w, scale)
        ref = fake_quant_reference(grouped, (amax_ref / 7.0).unsqueeze(-1), -8, 7).reshape(w.shape)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_per_group_indivisible_raises(self) -> None:
        q = IntQuantizer(4, "per_group", group_size=4)
        w = torch.randn(3, 10)
        with pytest.raises(ValueError, match="in_features=10.*group_size=4"):
            q.compute_scale(w)

    def test_per_group_requires_2d(self) -> None:
        q = IntQuantizer(4, "per_group", group_size=4)
        with pytest.raises(ValueError, match="2-D"):
            q.compute_scale(torch.randn(2, 3, 8))


class TestSTE:
    def test_backward_identity_in_range_zero_out_of_range(self) -> None:
        # frozen scale from a bigger calibration tensor; x contains an out-of-range entry
        calib = torch.tensor([[127.0, 50.0]])
        q = IntQuantizer(8, "per_channel")
        scale = q.compute_scale(calib)  # amax = 127 -> scale = 1.0
        x = torch.tensor([[3.0, -4.0, 200.0, -200.0]], requires_grad=True)
        out = q.quant_dequant(x, scale)
        grad = torch.autograd.grad(out.sum(), x)[0]
        ref_mask = x.detach().abs() <= 127.0
        assert torch.equal(grad, ref_mask.to(grad.dtype))
        assert grad[0, 0].item() == 1.0 and grad[0, 1].item() == 1.0
        assert grad[0, 2].item() == 0.0 and grad[0, 3].item() == 0.0

    def test_backward_per_token_mask(self) -> None:
        q = IntQuantizer(8, "per_token")
        calib = torch.ones(2, 3) * 127.0
        scale = q.compute_scale(calib)  # (2,1) scale = 1.0
        x = torch.tensor([[1.0, 2.0, 300.0], [4.0, -400.0, 5.0]], requires_grad=True)
        out = q.quant_dequant(x, scale)
        grad = torch.autograd.grad(out.sum(), x)[0]
        ref_mask = x.detach().abs() <= 127.0
        assert torch.equal(grad, ref_mask.to(grad.dtype))

    def test_forward_matches_no_autograd_node(self) -> None:
        q = IntQuantizer(4, "per_group", group_size=2)
        w = torch.randn(5, 6)
        scale = q.compute_scale(w)
        out = q.quant_dequant(w, scale)
        grouped = w.reshape(-1, 2)
        gscale = scale.flatten().unsqueeze(-1)
        ref = fake_quant_reference(grouped, gscale, -8, 7).reshape(w.shape)
        assert torch.equal(out, ref)
        x = torch.randn(4, 5)
        s_tok = IntQuantizer(8, "per_token")
        scale_a = s_tok.compute_scale(x)
        assert torch.equal(
            s_tok.quant_dequant(x, scale_a),
            fake_quant_reference(x, scale_a, -128, 127),
        )

    def test_scale_not_requiring_grad(self) -> None:
        q = IntQuantizer(8, "per_channel")
        x = torch.randn(3, 4, requires_grad=True)
        scale = q.compute_scale(x)
        assert not scale.requires_grad


class TestDtype:
    def test_output_dtype_float32(self) -> None:
        q = IntQuantizer(8, "per_channel")
        x = torch.randn(3, 4)
        out = q.quant_dequant(x, q.compute_scale(x))
        assert out.dtype == torch.float32

    def test_output_dtype_bfloat16(self) -> None:
        q = IntQuantizer(8, "per_token")
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        out = q.quant_dequant(x, q.compute_scale(x))
        assert out.dtype == torch.bfloat16
        ref = fake_quant_reference(x.float(), q.compute_scale(x), -128, 127).to(torch.bfloat16)
        assert torch.equal(out, ref)

    def test_output_dtype_bfloat16_per_group(self) -> None:
        q = IntQuantizer(4, "per_group", group_size=2)
        x = torch.randn(3, 4, dtype=torch.bfloat16)
        out = q.quant_dequant(x, q.compute_scale(x))
        assert out.dtype == torch.bfloat16


class TestConstruction:
    def test_invalid_bits_rejected(self) -> None:
        with pytest.raises(ValueError, match="bits=3"):
            IntQuantizer(3, "per_channel")  # type: ignore[arg-type]

    def test_unknown_scope_rejected(self) -> None:
        with pytest.raises(ValueError, match="scope"):
            IntQuantizer(8, "per_something")

    def test_group_scope_requires_group_size(self) -> None:
        with pytest.raises(ValueError, match="group_size"):
            IntQuantizer(4, "per_group", group_size=None)

    def test_non_group_scope_rejects_group_size(self) -> None:
        with pytest.raises(ValueError, match="group_size"):
            IntQuantizer(8, "per_channel", group_size=64)

    def test_dtype_and_qbounds(self) -> None:
        q8 = IntQuantizer(8, "per_channel")
        q4 = IntQuantizer(4, "per_token")
        assert q8.dtype == "int8" and q8.qmin == -128 and q8.qmax == 127
        assert q4.dtype == "int4" and q4.qmin == -8 and q4.qmax == 7

    def test_no_state_dict_leak(self) -> None:
        q = IntQuantizer(8, "per_channel")
        assert len(q.state_dict()) == 0
        assert sum(p.numel() for p in q.parameters()) == 0
        assert len(list(q.buffers())) == 0


class TestMsmodelslimGolden:
    """Golden对拍 against msmodelslim deploy IR numerics (design §4.3, §10.1).

    Skips cleanly when msmodelslim is not on PYTHONPATH.
    """

    def test_golden_fake_quant(self) -> None:
        api = pytest.importorskip("msmodelslim.core.quantization_schema.api")
        qal = pytest.importorskip("msmodelslim.core.quantization_schema.qal")
        calculate_qparam = api.calculate_qparam
        fake_quantize = api.fake_quantize
        q_dtype, q_scope, q_storage = qal.QDType, qal.QScope, qal.QStorage

        torch.manual_seed(0)
        q = IntQuantizer(8, "per_channel")
        w = torch.randn(4, 8)
        scale = q.compute_scale(w)
        out = q.quant_dequant(w, scale)
        # mirror the deploy IR call site (w8a8_dynamic.py): caller computes min/max,
        # calculate_qparam derives the symmetric scale, fake_quantize rounds+clamps+dequants.
        q_param = calculate_qparam(
            w.min(dim=1).values,
            w.max(dim=1).values,
            q_dtype.INT8,
            q_scope.PER_CHANNEL,
            True,
        )
        golden_scale = q_param.ext["scale"]
        assert torch.allclose(scale, golden_scale, atol=1e-6)
        # The IR stores weights transposed (in, out) so the per-output-channel scale
        # broadcasts along the last dim; mirror that call shape verbatim.
        golden_out = fake_quantize(q_storage(q_dtype.FLOAT, w.t()), q_param).value.t()
        assert torch.allclose(out, golden_out, atol=1e-6)
