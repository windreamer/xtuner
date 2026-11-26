# Copyright (c) OpenMMLab. All rights reserved.
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from xtuner.v1.quant.loss import (
    QADLossConfig,
    QADLossContext,
    QADLossKwargs,
    compute_qad_forward_kl,
)
from xtuner.v1.rl.loss.distillation_loss import compute_topk_distillation_kl


VOCAB = 128
TOP_K = 16
BATCH = 2
SEQ_LEN = 5


def _rand_logits(batch: int, seq: int, vocab: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, vocab, generator=generator, dtype=torch.float64)


def _random_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    student_logits = torch.randn(BATCH, SEQ_LEN, VOCAB, generator=generator, dtype=torch.float64)
    teacher_logits = torch.randn(BATCH, SEQ_LEN, VOCAB, generator=generator, dtype=torch.float64)
    loss_mask = (torch.rand(BATCH, SEQ_LEN, generator=generator) > 0.3).double()
    loss_mask[0, 0] = 1.0  # keep at least one valid position per test
    return student_logits, teacher_logits, loss_mask


def _teacher_topk_targets(teacher_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    target_token_ids = teacher_logits.topk(top_k, dim=-1).indices
    teacher_full_logprobs = torch.log_softmax(teacher_logits, dim=-1)
    teacher_topk_logprobs = teacher_full_logprobs.gather(dim=-1, index=target_token_ids)
    return target_token_ids, teacher_topk_logprobs


def _make_context(
    mode: str,
    loss_mask: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    target_token_ids: torch.Tensor | None = None,
    **config_overrides,
) -> QADLossContext:
    loss_cfg = QADLossConfig(mode=mode, **config_overrides)
    loss_kwargs = QADLossKwargs(
        loss_mask=loss_mask,
        teacher_logprobs=teacher_logprobs,
        target_token_ids=target_token_ids,
    )
    loss_ctx = QADLossContext(loss_cfg, loss_kwargs)
    QADLossContext.build_batches([loss_ctx])
    return loss_ctx


def _forward_loss(ctx: QADLossContext, student_logits: torch.Tensor) -> torch.Tensor:
    head_weight = torch.eye(VOCAB, dtype=student_logits.dtype)  # identity head: logits == hidden_states
    loss, _ = ctx.forward(student_logits, head_weight)
    return loss


def _masked_mean(per_token_loss: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    return (per_token_loss * loss_mask).sum() / loss_mask.sum()


class TestQADLossConfig:
    def test_defaults(self) -> None:
        cfg = QADLossConfig()
        assert cfg.mode == "qad_topk_forward"
        assert cfg.top_k == 64
        assert cfg.temperature == 1.0
        assert cfg.log_prob_min_clamp == -10.0
        assert cfg.loss_max_clamp == 10.0

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QADLossConfig(unknown_field=1)  # type: ignore[arg-type]

    def test_invalid_values_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QADLossConfig(top_k=0)
        with pytest.raises(ValidationError):
            QADLossConfig(temperature=0.0)
        with pytest.raises(ValidationError):
            QADLossConfig(loss_max_clamp=-1.0)

    def test_build_returns_none_without_teacher_targets(self) -> None:
        data = {"loss_mask": torch.ones(1, 4)}
        assert QADLossConfig().build(data, device="cpu") is None

    def test_build_topk_mode_requires_target_token_ids(self) -> None:
        data = {"loss_mask": torch.ones(1, 4), "teacher_logprobs": torch.zeros(1, 4, 8)}
        with pytest.raises(ValueError, match="target_token_ids"):
            QADLossConfig().build(data, device="cpu")

    def test_build_topk_mode(self) -> None:
        data = {
            "loss_mask": torch.ones(1, 4),
            "teacher_logprobs": torch.zeros(1, 4, 8),
            "target_token_ids": torch.zeros(1, 4, 8, dtype=torch.int64),
        }
        loss_ctx = QADLossConfig().build(data, device="cpu")
        assert isinstance(loss_ctx, QADLossContext)
        assert isinstance(loss_ctx.loss_kwargs, QADLossKwargs)

    def test_build_full_kl_mode(self) -> None:
        data = {"loss_mask": torch.ones(1, 4), "teacher_logprobs": torch.zeros(1, 4, 8)}
        loss_ctx = QADLossConfig(mode="qad_full_kl").build(data, device="cpu")
        assert isinstance(loss_ctx, QADLossContext)


class TestQADForwardKLKernel:
    def test_matches_opd_topk_forward(self) -> None:
        for seed in range(3):
            generator = torch.Generator().manual_seed(seed)
            student_logprobs = torch.log_softmax(torch.randn(4, 3, 16, generator=generator, dtype=torch.float64), -1)
            teacher_logprobs = torch.log_softmax(torch.randn(4, 3, 16, generator=generator, dtype=torch.float64), -1)
            ours = compute_qad_forward_kl(student_logprobs, teacher_logprobs, -10.0, 10.0)
            reference = compute_topk_distillation_kl(student_logprobs, teacher_logprobs, "forward", -10.0, 10.0)[0]
            assert torch.allclose(ours, reference, atol=1e-6)

    def test_tail_bin_hand_formula(self) -> None:
        student_logits = _rand_logits(BATCH, SEQ_LEN, VOCAB, seed=11)
        teacher_logits = _rand_logits(BATCH, SEQ_LEN, VOCAB, seed=12)
        target_token_ids, teacher_topk_logprobs = _teacher_topk_targets(teacher_logits, TOP_K)
        student_topk_logprobs = torch.log_softmax(student_logits, dim=-1).gather(dim=-1, index=target_token_ids)

        ours = compute_qad_forward_kl(student_topk_logprobs, teacher_topk_logprobs, -1e9, 1e9)

        teacher_probs = teacher_topk_logprobs.exp()
        student_probs = student_topk_logprobs.exp()
        teacher_tail = 1.0 - teacher_probs.sum(dim=-1)
        student_tail = 1.0 - student_probs.sum(dim=-1)
        expected = (teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)).sum(dim=-1)
        expected = expected + teacher_tail * (teacher_tail.log() - student_tail.log())
        assert torch.allclose(ours, expected, atol=1e-6)

    def test_loss_max_clamp_applied_to_per_token_loss(self) -> None:
        student_topk_logprobs = torch.tensor([[[-45.0, -20.0]]], dtype=torch.float64)
        teacher_topk_logprobs = torch.tensor([[[-0.1, -2.3]]], dtype=torch.float64)
        unclamped = compute_qad_forward_kl(student_topk_logprobs, teacher_topk_logprobs, -30.0, 1e9)
        assert unclamped.item() > 10.0
        clamped = compute_qad_forward_kl(student_topk_logprobs, teacher_topk_logprobs, -30.0, 10.0)
        assert clamped.item() == 10.0

    def test_min_clamp_keeps_loss_finite_and_matches_hand(self) -> None:
        student_logits, teacher_logits, loss_mask = _random_batch(seed=21)
        student_logits[..., 7] = -1e4  # student logprob far below log_prob_min_clamp
        target_token_ids, teacher_topk_logprobs = _teacher_topk_targets(teacher_logits, TOP_K)
        loss_ctx = _make_context(
            "qad_topk_forward", loss_mask, teacher_topk_logprobs, target_token_ids=target_token_ids
        )
        loss = _forward_loss(loss_ctx, student_logits)
        assert torch.isfinite(loss).all()

        student_topk_logprobs = torch.log_softmax(student_logits, dim=-1).gather(dim=-1, index=target_token_ids)
        # Clamp-first placement: probabilities, selected masses and tail bins are all derived
        # from the clamped log-probabilities.
        student_clamped = student_topk_logprobs.clamp_min(-10.0)
        teacher_clamped = teacher_topk_logprobs.clamp_min(-10.0)
        teacher_probs = teacher_clamped.exp()
        teacher_tail = 1.0 - teacher_probs.sum(dim=-1)
        student_tail = 1.0 - student_clamped.exp().sum(dim=-1)
        per_token = (teacher_probs * (teacher_clamped - student_clamped)).sum(-1)
        per_token = per_token + teacher_tail * (teacher_tail.log() - student_tail.log())
        assert torch.allclose(loss, _masked_mean(per_token, loss_mask), atol=1e-6)


class TestQADLossEquivalence:
    def test_topk_context_matches_opd_on_random_tensors(self) -> None:
        student_logits, teacher_logits, loss_mask = _random_batch(seed=31)
        target_token_ids, teacher_topk_logprobs = _teacher_topk_targets(teacher_logits, TOP_K)
        loss_ctx = _make_context("qad_topk_forward", loss_mask, teacher_topk_logprobs, target_token_ids)
        loss = _forward_loss(loss_ctx, student_logits)

        student_topk_logprobs = torch.log_softmax(student_logits, dim=-1).gather(dim=-1, index=target_token_ids)
        reference = compute_topk_distillation_kl(student_topk_logprobs, teacher_topk_logprobs, "forward", -10.0, 10.0)[
            0
        ]
        assert torch.allclose(loss, _masked_mean(reference, loss_mask), atol=1e-6)

    def test_full_kl_equals_topk_forward_with_k_equals_v(self) -> None:
        student_logits, teacher_logits, loss_mask = _random_batch(seed=32)
        teacher_full_logprobs = torch.log_softmax(teacher_logits, dim=-1)
        target_token_ids = torch.arange(VOCAB, dtype=torch.int64).expand(BATCH, SEQ_LEN, VOCAB)

        loss_full = _forward_loss(_make_context("qad_full_kl", loss_mask, teacher_full_logprobs), student_logits)
        loss_topk = _forward_loss(
            _make_context("qad_topk_forward", loss_mask, teacher_full_logprobs, target_token_ids), student_logits
        )
        assert torch.allclose(loss_full, loss_topk, atol=1e-6)

    def test_full_kl_matches_hand_kl_div(self) -> None:
        student_logits, teacher_logits, loss_mask = _random_batch(seed=33)
        teacher_full_logprobs = torch.log_softmax(teacher_logits, dim=-1)
        loss_ctx = _make_context("qad_full_kl", loss_mask, teacher_full_logprobs)
        loss = _forward_loss(loss_ctx, student_logits)

        per_token = F.kl_div(
            torch.log_softmax(student_logits, dim=-1),
            torch.softmax(teacher_logits, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        assert torch.allclose(loss, _masked_mean(per_token, loss_mask), atol=1e-6)


class TestQADLossTemperature:
    def test_full_kl_temperature_two_matches_hand(self) -> None:
        student_logits, teacher_logits, loss_mask = _random_batch(seed=41)
        temperature = 2.0
        teacher_logprobs = torch.log_softmax(teacher_logits / temperature, dim=-1)
        loss_ctx = _make_context("qad_full_kl", loss_mask, teacher_logprobs, temperature=temperature)
        loss = _forward_loss(loss_ctx, student_logits)

        per_token = F.kl_div(
            torch.log_softmax(student_logits / temperature, dim=-1),
            torch.softmax(teacher_logits / temperature, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        expected = (temperature**2) * _masked_mean(per_token, loss_mask)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_topk_temperature_two_matches_hand(self) -> None:
        student_logits, teacher_logits, loss_mask = _random_batch(seed=42)
        temperature = 2.0
        # Dividing logits by T>0 preserves the ranking, so the teacher top-K ids are unchanged.
        target_token_ids = teacher_logits.topk(TOP_K, dim=-1).indices
        teacher_topk_logprobs = torch.log_softmax(teacher_logits / temperature, dim=-1).gather(
            dim=-1, index=target_token_ids
        )
        loss_ctx = _make_context(
            "qad_topk_forward", loss_mask, teacher_topk_logprobs, target_token_ids, temperature=temperature
        )
        loss = _forward_loss(loss_ctx, student_logits)

        student_topk_logprobs = torch.log_softmax(student_logits / temperature, dim=-1).gather(
            dim=-1, index=target_token_ids
        )
        teacher_probs = teacher_topk_logprobs.exp()
        student_probs = student_topk_logprobs.exp()
        teacher_tail = 1.0 - teacher_probs.sum(dim=-1)
        student_tail = 1.0 - student_probs.sum(dim=-1)
        per_token = (teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)).sum(dim=-1)
        per_token = per_token + teacher_tail * (teacher_tail.log() - student_tail.log())
        expected = (temperature**2) * _masked_mean(per_token, loss_mask)
        assert torch.allclose(loss, expected, atol=1e-6)


class TestQADLossReduction:
    def test_mask_single_position(self) -> None:
        student_logits, teacher_logits, _ = _random_batch(seed=51)
        loss_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.float64)
        loss_mask[1, 3] = 1.0
        teacher_full_logprobs = torch.log_softmax(teacher_logits, dim=-1)
        loss_ctx = _make_context("qad_full_kl", loss_mask, teacher_full_logprobs)
        loss = _forward_loss(loss_ctx, student_logits)

        per_token = F.kl_div(
            torch.log_softmax(student_logits, dim=-1),
            torch.softmax(teacher_logits, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        assert torch.allclose(loss, per_token[1, 3], atol=1e-6)

    def test_empty_mask_returns_zero(self) -> None:
        student_logits, teacher_logits, _ = _random_batch(seed=52)
        loss_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.float64)
        teacher_full_logprobs = torch.log_softmax(teacher_logits, dim=-1)
        loss_ctx = _make_context("qad_full_kl", loss_mask, teacher_full_logprobs)
        loss = _forward_loss(loss_ctx, student_logits)
        assert loss.item() == 0.0

    def test_build_batches_calibrates_across_microbatches(self) -> None:
        student_logits_1, teacher_logits_1, loss_mask_1 = _random_batch(seed=61)
        student_logits_2, teacher_logits_2, loss_mask_2 = _random_batch(seed=62)
        teacher_logprobs_1 = torch.log_softmax(teacher_logits_1, dim=-1)
        teacher_logprobs_2 = torch.log_softmax(teacher_logits_2, dim=-1)

        loss_cfg = QADLossConfig(mode="qad_full_kl")
        ctx_1 = QADLossContext(loss_cfg, QADLossKwargs(loss_mask=loss_mask_1, teacher_logprobs=teacher_logprobs_1))
        ctx_2 = QADLossContext(loss_cfg, QADLossKwargs(loss_mask=loss_mask_2, teacher_logprobs=teacher_logprobs_2))
        QADLossContext.build_batches([ctx_1, ctx_2])
        loss_1 = _forward_loss(ctx_1, student_logits_1)
        loss_2 = _forward_loss(ctx_2, student_logits_2)

        # The step loss equals the global masked mean over both micro-batches.
        per_token_1 = F.kl_div(
            torch.log_softmax(student_logits_1, dim=-1), torch.softmax(teacher_logits_1, dim=-1), reduction="none"
        ).sum(dim=-1)
        per_token_2 = F.kl_div(
            torch.log_softmax(student_logits_2, dim=-1), torch.softmax(teacher_logits_2, dim=-1), reduction="none"
        ).sum(dim=-1)
        global_mask_sum = loss_mask_1.sum() + loss_mask_2.sum()
        expected = (per_token_1 * loss_mask_1).sum() / global_mask_sum
        expected = expected + (per_token_2 * loss_mask_2).sum() / global_mask_sum
        assert torch.allclose(loss_1 + loss_2, expected, atol=1e-6)

        # The calibrated loss also matches a single context over the concatenated micro-batches.
        full_kwargs = cast(
            QADLossKwargs,
            QADLossKwargs.cat(
                [
                    QADLossKwargs(loss_mask=loss_mask_1, teacher_logprobs=teacher_logprobs_1),
                    QADLossKwargs(loss_mask=loss_mask_2, teacher_logprobs=teacher_logprobs_2),
                ]
            ),
        )
        ctx_full = QADLossContext(loss_cfg, full_kwargs)
        QADLossContext.build_batches([ctx_full])
        loss_full = _forward_loss(ctx_full, torch.cat([student_logits_1, student_logits_2], dim=1))
        assert torch.allclose(loss_full, loss_1 + loss_2, atol=1e-6)


def _coupling_logits(seed: int) -> torch.Tensor:
    """Moderate-magnitude logits: log-softmax stays above ``log_prob_min_clamp`` and the
    per-token KL stays below ``loss_max_clamp``, so neither clamp binds and the analytic
    gradient comparison exercises the raw coupling."""
    generator = torch.Generator().manual_seed(seed)
    return 0.5 * torch.randn(BATCH, SEQ_LEN, VOCAB, generator=generator, dtype=torch.float64)


def _coupling_mask(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    loss_mask = (torch.rand(BATCH, SEQ_LEN, generator=generator) > 0.3).double()
    loss_mask[0, 0] = 1.0  # keep at least one valid position
    return loss_mask


class TestQADGradientCoupling:
    """Regression guard for Gate B: the loss context's autograd gradient w.r.t. the student
    logits must equal the analytic forward-KL gradient. A sign flip, reverse-KL, mask/shift
    misalignment or detached branch makes these tests fail."""

    def test_full_kl_gradient_matches_analytic(self) -> None:
        student_logits = _coupling_logits(seed=71).requires_grad_(True)
        teacher_logits = _coupling_logits(seed=72)
        loss_mask = _coupling_mask(seed=73)
        loss_ctx = _make_context("qad_full_kl", loss_mask, torch.log_softmax(teacher_logits, dim=-1))

        loss = _forward_loss(loss_ctx, student_logits)
        (grad,) = torch.autograd.grad(loss, student_logits)

        # d/ds_v of mask/N * sum_v' q_v' (log q_v' - log p_v') is mask/N * (p_v - q_v).
        probs_student = torch.softmax(student_logits, dim=-1)
        probs_teacher = torch.softmax(teacher_logits, dim=-1)
        expected = (loss_mask / loss_mask.sum()).unsqueeze(-1) * (probs_student - probs_teacher)
        assert torch.allclose(grad, expected, atol=1e-6)
        # Masked positions must receive exactly zero gradient.
        assert bool((grad[loss_mask == 0] == 0).all())

    def test_topk_gradient_matches_analytic_tail_bin_form(self) -> None:
        student_logits = _coupling_logits(seed=81).requires_grad_(True)
        teacher_logits = _coupling_logits(seed=82)
        loss_mask = _coupling_mask(seed=83)
        target_token_ids, teacher_topk_logprobs = _teacher_topk_targets(teacher_logits, TOP_K)
        loss_ctx = _make_context("qad_topk_forward", loss_mask, teacher_topk_logprobs, target_token_ids)

        loss = _forward_loss(loss_ctx, student_logits)
        (grad,) = torch.autograd.grad(loss, student_logits)

        # Truncated KL = sum_{k in S} q_k (log q_k - log p_k) + q_tail (log q_tail - log p_tail)
        # with p/q the softmax masses, p_tail = 1 - p_S, q_tail = 1 - q_S. Its exact gradient:
        #   d/ds_v = p_v * (q_S - q_tail * (p_S - I[v in S]) / p_tail) - q_v * I[v in S].
        probs_student = torch.softmax(student_logits, dim=-1)
        probs_teacher = torch.softmax(teacher_logits, dim=-1)
        q_selected = probs_teacher.gather(-1, index=target_token_ids)
        q_selected_mass = q_selected.sum(dim=-1, keepdim=True)
        q_tail = 1.0 - q_selected_mass
        p_selected_mass = probs_student.gather(-1, index=target_token_ids).sum(dim=-1, keepdim=True)
        p_tail = 1.0 - p_selected_mass
        in_support = torch.zeros_like(probs_student)
        in_support.scatter_(-1, target_token_ids, 1.0)
        expected_per_token = probs_student * (q_selected_mass - q_tail * (p_selected_mass - in_support) / p_tail)
        expected_per_token = expected_per_token - probs_teacher * in_support
        expected = (loss_mask / loss_mask.sum()).unsqueeze(-1) * expected_per_token
        assert torch.allclose(grad, expected, atol=1e-6)

    def test_topk_full_support_gradient_equals_full_kl(self) -> None:
        student_logits = _coupling_logits(seed=91)
        teacher_logits = _coupling_logits(seed=92)
        loss_mask = _coupling_mask(seed=93)
        # K = V: the teacher support is a permutation of the whole vocabulary, so both tail
        # bins vanish and the truncated kernel must reduce to the exact full-KL coupling.
        target_token_ids, teacher_topk_logprobs = _teacher_topk_targets(teacher_logits, VOCAB)
        ctx_topk = _make_context("qad_topk_forward", loss_mask, teacher_topk_logprobs, target_token_ids)
        ctx_full = _make_context("qad_full_kl", loss_mask, torch.log_softmax(teacher_logits, dim=-1))

        s_topk = student_logits.clone().requires_grad_(True)
        loss_topk = _forward_loss(ctx_topk, s_topk)
        grad_topk = torch.autograd.grad(loss_topk, s_topk)[0]
        s_full = student_logits.clone().requires_grad_(True)
        loss_full = _forward_loss(ctx_full, s_full)
        grad_full = torch.autograd.grad(loss_full, s_full)[0]

        assert torch.allclose(loss_topk, loss_full, atol=1e-6)
        assert torch.allclose(grad_topk, grad_full, atol=1e-6)
