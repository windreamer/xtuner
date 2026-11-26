# Copyright (c) OpenMMLab. All rights reserved.
"""Tests of the QAD teacher adapter (design doc §7.1/§7.2 target contracts)."""

import pytest
import torch

from xtuner.v1.model.base import MHAConfig, TransformerConfig
from xtuner.v1.quant.loss import QADLossConfig, QADLossContext
from xtuner.v1.quant.teacher import QADTeacherAdapter, QADTeacherAdapterConfig
from xtuner.v1.rl.distillation.config import TrainTeacherConfig


VOCAB = 32
TOP_K = 8
BATCH = 2
SEQ_LEN = 5
CHUNK_SIZE = 2  # smaller than SEQ_LEN so chunk boundaries are exercised


class _StubLMHead(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.record: list[tuple[int, ...]] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.linear(hidden_states)
        self.record.append(tuple(logits.shape))
        return logits


class _StubTeacher(torch.nn.Module):
    """Embedding + one position-wise layer + lm_head.

    The layer mixes no information across positions, so the teacher is causal by construction and a sequence slice
    reproduces the full-context logits at the same positions (the chunked-forward assumption of the adapter).
    """

    def __init__(self, vocab_size: int = VOCAB, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.lm_head = _StubLMHead(hidden_size, vocab_size)
        self.input_record: list[tuple[int, int]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.input_record.append(tuple(input_ids.shape))
        hidden = self.embedding(input_ids)
        if position_ids is not None:
            hidden = hidden + position_ids.unsqueeze(-1).to(hidden.dtype)
        return self.lm_head(self.layer(hidden))


def _teacher_model_cfg(vocab_size: int = VOCAB) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=vocab_size,
        max_position_embeddings=128,
        eos_token_id=0,
        num_hidden_layers=1,
        hidden_size=8,
        intermediate_size=16,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=2, num_key_value_heads=2, head_dim=4),
    )


def _make_adapter(
    student_vocab_size: int | TransformerConfig | None = None,
    teacher_offload: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> QADTeacherAdapter:
    config = QADTeacherAdapterConfig(
        teacher=TrainTeacherConfig(name="teacher", model_path="stub", model_cfg=_teacher_model_cfg()),
        teacher_offload=teacher_offload,
        chunk_size=chunk_size,
    )
    resolved_student = VOCAB if student_vocab_size is None else student_vocab_size
    return QADTeacherAdapter(config, resolved_student)


def _make_batch(
    seed: int = 0,
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    vocab: int = VOCAB,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len).contiguous()
    return {
        "input_ids": torch.randint(0, vocab, (batch, seq_len), generator=generator),
        "position_ids": position_ids,
        # Extra field the adapter must ignore (not forwarded to the teacher forward).
        "loss_mask": (torch.rand(batch, seq_len, generator=generator) > 0.3).float(),
    }


def _full_logits(teacher: _StubTeacher, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return teacher(input_ids=batch["input_ids"], position_ids=batch["position_ids"])


def _reference_logprobs(
    teacher: _StubTeacher,
    batch: dict[str, torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    return torch.log_softmax(_full_logits(teacher, batch).float() / temperature, dim=-1)


class TestQADTeacherTopK:
    def test_topk_contract(self) -> None:
        adapter = _make_adapter()
        teacher = _StubTeacher()
        batch = _make_batch()
        loss_cfg = QADLossConfig(mode="qad_topk_forward", top_k=TOP_K)

        targets = adapter.compute_targets(teacher, batch, loss_cfg)

        ids = targets["target_token_ids"]
        logprobs = targets["teacher_logprobs"]
        assert ids.shape == (BATCH, SEQ_LEN, TOP_K)
        assert ids.dtype == torch.int64
        assert logprobs.shape == (BATCH, SEQ_LEN, TOP_K)
        assert logprobs.dtype == torch.float32
        expected_ids = (_full_logits(teacher, batch).float() / loss_cfg.temperature).topk(TOP_K, dim=-1).indices
        assert torch.equal(ids, expected_ids)
        expected_logprobs = _reference_logprobs(teacher, batch, loss_cfg.temperature).gather(dim=-1, index=ids)
        assert torch.allclose(logprobs, expected_logprobs, atol=1e-6)

    def test_topk_temperature_tempering(self) -> None:
        adapter = _make_adapter()
        teacher = _StubTeacher()
        batch = _make_batch()
        loss_cfg = QADLossConfig(mode="qad_topk_forward", top_k=TOP_K, temperature=2.0)

        tempered = adapter.compute_targets(teacher, batch, loss_cfg)
        untempered = adapter.compute_targets(teacher, batch, QADLossConfig(mode="qad_topk_forward", top_k=TOP_K))

        # Tempering preserves ordering: same ids, tempered values.
        assert torch.equal(tempered["target_token_ids"], untempered["target_token_ids"])
        assert not torch.allclose(tempered["teacher_logprobs"], untempered["teacher_logprobs"], atol=1e-6)
        expected = _reference_logprobs(teacher, batch, 2.0).gather(dim=-1, index=tempered["target_token_ids"])
        assert torch.allclose(tempered["teacher_logprobs"], expected, atol=1e-6)

    def test_bf16_teacher_targets_are_fp32_upcast(self) -> None:
        adapter = _make_adapter()
        teacher = _StubTeacher().to(torch.bfloat16)
        batch = _make_batch()
        loss_cfg = QADLossConfig(mode="qad_topk_forward", top_k=TOP_K, temperature=2.0)

        targets = adapter.compute_targets(teacher, batch, loss_cfg)

        logprobs = targets["teacher_logprobs"]
        assert logprobs.dtype == torch.float32
        expected = _reference_logprobs(teacher, batch, 2.0).gather(dim=-1, index=targets["target_token_ids"])
        assert torch.allclose(logprobs, expected, atol=1e-6)

    def test_missing_input_ids_raises(self) -> None:
        adapter = _make_adapter()
        batch = _make_batch()
        del batch["input_ids"]
        with pytest.raises(ValueError, match="input_ids"):
            adapter.compute_targets(_StubTeacher(), batch, QADLossConfig(mode="qad_topk_forward", top_k=TOP_K))


class TestQADTeacherFullKL:
    def test_full_kl_contract_and_chunk_boundaries(self) -> None:
        adapter = _make_adapter(chunk_size=CHUNK_SIZE)
        teacher = _StubTeacher()
        batch = _make_batch()
        loss_cfg = QADLossConfig(mode="qad_full_kl")

        targets = adapter.compute_targets(teacher, batch, loss_cfg)

        assert set(targets) == {"teacher_logprobs"}
        logprobs = targets["teacher_logprobs"]
        assert logprobs.shape == (BATCH, SEQ_LEN, VOCAB)
        assert logprobs.dtype == torch.float32
        reference = torch.log_softmax(_full_logits(teacher, batch).float(), dim=-1)
        assert torch.allclose(logprobs, reference, atol=1e-6)

    def test_chunked_forward_never_materializes_full_logits(self) -> None:
        seq_len = 6
        chunk_size = 4
        adapter = _make_adapter(chunk_size=chunk_size)
        teacher = _StubTeacher()
        batch = _make_batch(batch=1, seq_len=seq_len)

        targets = adapter.compute_targets(teacher, batch, QADLossConfig(mode="qad_full_kl"))

        assert targets["teacher_logprobs"].shape == (1, seq_len, VOCAB)
        # The lm_head records every logits tensor it produced; the chunked path must never emit
        # more than one chunk of rows, i.e. no [1, seq_len, V] logits tensor is ever built.
        recorded = teacher.lm_head.record
        assert recorded == [(1, chunk_size, VOCAB), (1, seq_len - chunk_size, VOCAB)]
        assert max(shape[0] * shape[1] for shape in recorded) <= chunk_size


class TestQADTeacherGrad:
    def test_targets_detached_and_teacher_params_untouched(self) -> None:
        adapter = _make_adapter()
        teacher = _StubTeacher()
        params_before = {name: param.detach().clone() for name, param in teacher.named_parameters()}
        batch = _make_batch()
        loss_cfg = QADLossConfig(mode="qad_topk_forward", top_k=TOP_K)

        targets = adapter.compute_targets(teacher, batch, loss_cfg)
        assert teacher.training  # eval mode is only enforced for the call itself

        for tensor in targets.values():
            assert not tensor.requires_grad
        loss_ctx = loss_cfg.build({**batch, **targets}, device="cpu")
        assert loss_ctx is not None
        QADLossContext.build_batches([loss_ctx])
        hidden = torch.randn(BATCH, SEQ_LEN, VOCAB, requires_grad=True)
        head_weight = torch.eye(VOCAB, requires_grad=True)
        loss, _ = loss_ctx.forward(hidden, head_weight)
        loss.backward()

        for name, param in teacher.named_parameters():
            assert param.grad is None
            assert torch.equal(param.detach(), params_before[name])
        assert head_weight.grad is not None


class TestQADTeacherConstruction:
    def test_vocab_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            _make_adapter(student_vocab_size=VOCAB + 1)

    def test_vocab_from_student_model_config(self) -> None:
        adapter = _make_adapter(student_vocab_size=_teacher_model_cfg())
        assert adapter.teacher_vocab_size == VOCAB


class TestQADTeacherOffload:
    def test_offload_hooks_fire_around_compute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _make_adapter(teacher_offload=True)
        calls: list[str] = []
        monkeypatch.setattr(adapter, "_onload", lambda model: calls.append("onload"))
        monkeypatch.setattr(adapter, "_offload", lambda model: calls.append("offload"))

        adapter.compute_targets(_StubTeacher(), _make_batch(), QADLossConfig(mode="qad_full_kl"))

        assert calls == ["onload", "offload"]

    def test_offload_disabled_skips_hooks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _make_adapter(teacher_offload=False)
        calls: list[str] = []
        monkeypatch.setattr(adapter, "_onload", lambda model: calls.append("onload"))
        monkeypatch.setattr(adapter, "_offload", lambda model: calls.append("offload"))

        adapter.compute_targets(_StubTeacher(), _make_batch(), QADLossConfig(mode="qad_full_kl"))

        assert calls == []

    def test_timings_record_offload_phases(self) -> None:
        adapter = _make_adapter(teacher_offload=False)
        adapter.compute_targets(_StubTeacher(), _make_batch(), QADLossConfig(mode="qad_full_kl"))

        assert adapter.last_timings.onload == 0.0
        assert adapter.last_timings.offload == 0.0
        assert adapter.last_timings.compute > 0.0
