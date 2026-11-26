"""Model-Optimizer pattern-port tests for the QAD quantization package.

Ports the test methodology of NVIDIA Model-Optimizer (read-only reference under
``/data/windreamer/codebase/Model-Optimizer``) onto the landed QAD package:

- Pattern A — STE backward identity + clip-gradient mask
  (``tests/_test_utils/torch/quantization/tensor_quant_common.py``, ``test_backward`` /
  ``test_clip_gradient``).
- Pattern B — independent-reference numerics: a second fake-quant implementation sharing no code
  with :class:`IntQuantizer`; a red here is a real numerics bug.
- Pattern C — KD oracle zero loss, temperature hand oracle and config validation
  (``tests/unit/torch/distill/plugins/test_huggingface_kd.py``, ``_manual_kd_loss``).
- Pattern E — cached-scale step equals uncached recomputation plus the cache staleness contract
  (``tests/unit/torch/quantization/test_weight_cache.py``).

All tests are CPU-only and self-contained (seeded generators, no hub downloads).
"""

import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.quant.config import QADQuantConfig
from xtuner.v1.quant.fake_quant_linear import FakeQuantLinear
from xtuner.v1.quant.loss import QADLossConfig, QADLossContext, QADLossKwargs
from xtuner.v1.quant.quantizer import IntQuantizer


VOCAB = 64
TOP_K = 16
BATCH = 2
SEQ_LEN = 5

# Tie-dodging seed (Pattern B): under 123 no x/scale lands within 1e-3 of a .5 rounding boundary,
# so the where-based tie rule of the independent reference cannot be flipped by 1-ulp fp noise.
# The precondition is asserted in-test by ``_assert_no_rounding_ties``.
TIE_DODGE_SEED = 123


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _reference_fake_quant(x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """Second fake-quant reference sharing no code with :class:`IntQuantizer`.

    Round-half-even is built explicitly from floor/fract with a ``torch.where`` tie rule
    (IntQuantizer uses ``torch.round``), then clamped and rescaled — the deploy-IR contract.
    A mismatch against IntQuantizer is therefore a real numerics bug, not a shared-code artifact.

    Args:
        x (torch.Tensor): Incoming tensor; quantized per leading axis against ``scale``.
        scale (torch.Tensor): fp32 scale broadcastable to ``x``.
        qmin (int): Lower quantization level (inclusive).
        qmax (int): Upper quantization level (inclusive).

    Returns:
        torch.Tensor: Dequantized tensor in ``x``'s dtype.
    """
    y = x.detach().float() / scale
    floor = torch.floor(y)
    frac = y - floor
    floor_is_even = torch.remainder(floor, 2.0) == 0.0
    round_up = (frac > 0.5) | ((frac == 0.5) & ~floor_is_even)
    quantized = torch.where(round_up, floor + 1.0, floor).clamp(min=float(qmin), max=float(qmax))
    return (quantized * scale).to(x.dtype)


def _assert_no_rounding_ties(y: torch.Tensor) -> None:
    """Assert the tie-dodge precondition of seed 123 (see module docstring).

    Args:
        y (torch.Tensor): The pre-rounding quotients ``x / scale``.
    """
    frac2 = y * 2.0 - torch.floor(y * 2.0)
    assert torch.all((frac2 - 0.5).abs() > 1e-3), "seed no longer dodges .5 rounding boundaries"


def _make_ctx(
    mode: str,
    loss_mask: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    target_token_ids: torch.Tensor | None = None,
    **overrides,
) -> QADLossContext:
    """Build a calibrated :class:`QADLossContext` for the given teacher
    targets."""
    loss_cfg = QADLossConfig(mode=mode, **overrides)
    loss_kwargs = QADLossKwargs(
        loss_mask=loss_mask, teacher_logprobs=teacher_logprobs, target_token_ids=target_token_ids
    )
    loss_ctx = QADLossContext(loss_cfg, loss_kwargs)
    QADLossContext.build_batches([loss_ctx])
    return loss_ctx


def _forward_loss(ctx: QADLossContext, student_logits: torch.Tensor) -> torch.Tensor:
    """Run the loss context through an identity head so ``hidden_states ==
    logits``."""
    head_weight = torch.eye(VOCAB, dtype=student_logits.dtype)
    loss, _ = ctx.forward(student_logits, head_weight)
    return loss


class TestSteClipGradient:
    """Pattern A port: STE backward identity and clip mask
    (tensor_quant_common.py ``test_backward`` / ``test_clip_gradient``)."""

    def _clipped_setup(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Crafted input, scale shrunk to 0.7x amax (forces clipping), grad
        output and in-range mask."""
        x = torch.randn(4, 9, generator=_generator(TIE_DODGE_SEED)) * 1.5
        scale = IntQuantizer(8, "per_channel").compute_scale(x) * 0.7
        grad_output = torch.randn(x.shape, generator=_generator(TIE_DODGE_SEED + 1))
        in_range = x.abs() <= scale.unsqueeze(-1) * 127  # STE mask: |x| <= scale * qmax
        return x, scale, grad_output, in_range

    def test_clip_gradient_mask_exact(self) -> None:
        """(x.grad != 0) equals the in-range mask exactly; grads are the
        upstream grad masked."""
        x0, scale, grad_output, in_range = self._clipped_setup()
        # Both regions must exist or the mask assertions bite nothing.
        assert in_range.any() and not in_range.all()

        x = x0.clone().requires_grad_(True)
        out = IntQuantizer(8, "per_channel").quant_dequant(x, scale)
        (out * grad_output).sum().backward()

        assert x.grad is not None
        # Boolean clip mask must match exactly (ModelOpt: torch.allclose(x.grad != 0, x_in_range)).
        assert torch.equal(x.grad != 0, in_range)
        # Upstream-grad identity: d(out)/dx equals grad_output wherever |x| is in range.
        assert torch.equal(x.grad, grad_output * in_range.to(grad_output.dtype))
        assert torch.allclose(x.grad[in_range], grad_output[in_range])

    def test_downstream_nonlinear_scalar_grad_is_identity_passthrough(self) -> None:
        """ModelOpt ``test_backward`` form: with a nonlinear downstream scalar,
        the input grad equals the grad at the quant-node boundary (identity
        passthrough) in range."""
        x0, scale, _grad_output, in_range = self._clipped_setup()
        labels = torch.randint(0, 9, (4,), generator=_generator(TIE_DODGE_SEED + 2))

        x = x0.clone().requires_grad_(True)
        out = IntQuantizer(8, "per_channel").quant_dequant(x, scale)
        F.cross_entropy(out, labels).backward()

        # Reference: the same downstream scalar applied at the quant-node boundary, as if the
        # quantizer were an identity (the STE claim), as a leaf tensor.
        out_ref = out.detach().requires_grad_(True)
        F.cross_entropy(out_ref, labels).backward()

        assert x.grad is not None and out_ref.grad is not None
        assert torch.allclose(x.grad[in_range], out_ref.grad[in_range])
        assert torch.equal(x.grad[~in_range], torch.zeros_like(x.grad[~in_range]))


class TestIndependentReferenceNumerics:
    """Pattern B port: IntQuantizer numerics against an independent reference.

    If these go red, it is a real numerics bug — not a shared-code artifact.
    """

    def test_per_channel_int8_with_clipping(self) -> None:
        generator = _generator(TIE_DODGE_SEED)
        x = torch.randn(4, 9, generator=generator) * 1.5
        q = IntQuantizer(8, "per_channel")
        scale = q.compute_scale(x) * 0.7  # amax shrunk 0.7x: exercises the clamp path too
        scale_col = scale.unsqueeze(-1)  # (4, 1): broadcast over the trailing dim
        _assert_no_rounding_ties(x.float() / scale_col)

        ours = q.quant_dequant(x, scale)
        reference = _reference_fake_quant(x, scale_col, -128, 127)
        assert torch.allclose(ours, reference, atol=1e-6)

    def test_per_group_int4_with_clipping(self) -> None:
        generator = _generator(TIE_DODGE_SEED)
        w = torch.randn(3, 8, generator=generator) * 2.0
        q4 = IntQuantizer(4, "per_group", group_size=4)
        scale = q4.compute_scale(w) * 0.7
        grouped = w.reshape(3, 2, 4)
        _assert_no_rounding_ties(grouped.float() / scale.reshape(3, 2, 1))

        ours = q4.quant_dequant(w, scale)
        reference = _reference_fake_quant(grouped, scale.reshape(3, 2, 1), -8, 7).reshape(w.shape)
        assert torch.allclose(ours, reference, atol=1e-6)


class TestKDLossOracle:
    """Pattern C port: KD zero-loss oracle and temperature hand oracle
    (test_huggingface_kd.py ``_manual_kd_loss``)."""

    def _random_logits(self, seed: int) -> torch.Tensor:
        return torch.randn(BATCH, SEQ_LEN, VOCAB, generator=_generator(seed), dtype=torch.float64)

    def _mask_with_a_skipped_position(self) -> torch.Tensor:
        loss_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.float64)
        loss_mask[0, 0] = 0.0
        return loss_mask

    def test_identical_logits_zero_loss_full_kl(self) -> None:
        logits = self._random_logits(41)
        teacher_logprobs = torch.log_softmax(logits, dim=-1)
        ctx = _make_ctx("qad_full_kl", self._mask_with_a_skipped_position(), teacher_logprobs)
        loss = _forward_loss(ctx, logits)
        assert abs(loss.item()) < 1e-5

    def test_identical_logits_zero_loss_topk(self) -> None:
        # K < V: identical distributions select identical top-K, so the tail bin vanishes on both
        # sides and the loss must still be zero.
        logits = self._random_logits(42)
        target_token_ids = logits.topk(TOP_K, dim=-1).indices
        teacher_logprobs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=target_token_ids)
        ctx = _make_ctx(
            "qad_topk_forward",
            self._mask_with_a_skipped_position(),
            teacher_logprobs,
            target_token_ids=target_token_ids,
        )
        loss = _forward_loss(ctx, logits)
        assert abs(loss.item()) < 1e-5

    def test_temperature_two_hand_oracle(self) -> None:
        """_manual_kd_loss port: T=2 hand-computed KL with masked mean and T^2
        scaling."""
        temperature = 2.0
        student_logits = self._random_logits(44)
        teacher_logits = self._random_logits(45)
        loss_mask = self._mask_with_a_skipped_position()
        teacher_logprobs = torch.log_softmax(teacher_logits / temperature, dim=-1)
        ctx = _make_ctx("qad_full_kl", loss_mask, teacher_logprobs, temperature=temperature)
        loss = _forward_loss(ctx, student_logits)

        # Hand oracle, computed independently of the loss kernel.
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
        student_logprobs = torch.log_softmax(student_logits / temperature, dim=-1)
        per_token = (teacher_probs * (teacher_probs.log() - student_logprobs)).sum(dim=-1)
        expected = (temperature**2) * (per_token * loss_mask).sum() / loss_mask.sum()
        assert loss.item() == pytest.approx(expected.item())


class TestQADLossConfigValidation:
    """Pattern C config guard: invalid top_k / temperature raise ValueError at
    build time."""

    def test_invalid_top_k_raises_value_error(self) -> None:
        # pydantic's ValidationError subclasses ValueError, so ValueError-based handling works.
        with pytest.raises(ValueError, match="top_k"):
            QADLossConfig(top_k=0)
        with pytest.raises(ValueError, match="top_k"):
            QADLossConfig(top_k=-3)

    def test_invalid_temperature_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            QADLossConfig(temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            QADLossConfig(temperature=-1.0)


class TestWeightCacheEquivalence:
    """Pattern E port: cached-scale step matches uncached recomputation and the
    staleness contract holds (test_weight_cache.py
    ``test_cached_step_matches_uncached``)."""

    def _make_layer(self) -> FakeQuantLinear:
        torch.manual_seed(TIE_DODGE_SEED)
        return FakeQuantLinear(9, 5, quant_cfg=QADQuantConfig(quant_schema="w8a8"))

    def _uncached_reference(self, source: FakeQuantLinear) -> FakeQuantLinear:
        """Same weights, but the weight-scale cache is invalidated before every
        forward."""
        ref = FakeQuantLinear(
            source.in_features, source.out_features, bias=source.bias is not None, quant_cfg=QADQuantConfig()
        )
        with torch.no_grad():
            ref.weight.copy_(source.weight)
            if source.bias is not None:
                ref.bias.copy_(source.bias)
        ref._w_scale_cache = None
        return ref

    def test_cached_step_matches_uncached(self) -> None:
        lin = self._make_layer()
        lin.begin_step()
        x1 = torch.randn(3, 9, generator=_generator(1)).requires_grad_(True)
        out1 = lin(x1)
        x2 = torch.randn(4, 9, generator=_generator(2))
        out2 = lin(x2)  # second forward within the step reuses the cached scale

        ref = self._uncached_reference(lin)
        x_ref = x1.detach().clone().requires_grad_(True)
        ref_out1 = ref(x_ref)
        ref._w_scale_cache = None
        ref_out2 = ref(x2)

        assert torch.equal(out1, ref_out1)
        assert torch.equal(out2, ref_out2)

        out1.sum().backward()
        ref_out1.sum().backward()
        assert x1.grad is not None and x_ref.grad is not None
        assert torch.equal(x1.grad, x_ref.grad)

    def test_staleness_contract_within_step(self) -> None:
        lin = self._make_layer()
        lin.begin_step()
        x = torch.randn(3, 9, generator=_generator(3))
        out_pre = lin(x)
        assert lin._w_scale_cache is not None
        cached_scale = lin._w_scale_cache.clone()
        # Cache: two forwards within one step are identical.
        assert torch.equal(out_pre, lin(x))

        # Documented staleness contract: in-place weight mutation without begin_step keeps the
        # cached scale but re-quantizes the new weight with it, so the output changes.
        with torch.no_grad():
            lin.weight.mul_(0.5)
        assert torch.equal(lin._w_scale_cache, cached_scale)
        assert not torch.equal(out_pre, lin(x))

        # begin_step invalidates the cache: the output reflects the new weights.
        lin.begin_step()
        wq = IntQuantizer(8, "per_channel")
        xq = IntQuantizer(8, "per_token")
        w = lin.weight.detach()
        expected = F.linear(
            xq.quant_dequant(x, xq.compute_scale(x)),
            wq.quant_dequant(w, wq.compute_scale(w)),
            lin.bias,
        )
        assert torch.equal(lin(x), expected)
