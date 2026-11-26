# Copyright (c) OpenMMLab. All rights reserved.
"""Tests of the QAD trainer integration (design doc §8, §10.1
``TestQADTrainer``).

The tiny student/teacher model is constructed locally (``Qwen3DenseConfig``, 2 layers, hidden 64, vocab 128) rather
than downloaded from a HF hub, so every test is self-contained. The single-process tests cover config validation, the
teacher-target seam and the scale-freeze step arithmetic; the end-to-end one-step runs drive the real ``Trainer``/FSDP
pipeline and therefore need the training device (the base ``Trainer._init_dist`` has no CPU backend path), matching the
rest of ``tests/train``.
"""

import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from pydantic import ValidationError
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.datasets.config import BaseDataloaderConfig
from xtuner.v1.model.dense.qwen3 import Qwen3DenseConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.quant.config import KVQConfig, QADQuantConfig, SelectiveQuantConfig
from xtuner.v1.quant.fake_quant_linear import FakeQuantLinear
from xtuner.v1.quant.kv_cache import apply_kv_quantization
from xtuner.v1.quant.loss import QADLossConfig
from xtuner.v1.quant.teacher import QADTeacherAdapter, QADTeacherAdapterConfig
from xtuner.v1.rl.distillation.config import TrainTeacherConfig
from xtuner.v1.train.qad_trainer import (
    HoldoutConfig,
    QADTrainer,
    QADTrainerConfig,
    _loss_mask,
    _QADTeacherForward,
    resolve_qad_trainer_config,
)
from xtuner.v1.train.trainer import HooksConfig, TrainerConfig
from xtuner.v1.utils.device import get_device


DEVICE = get_device()

VOCAB = 128
HIDDEN = 64
LAYERS = 2
SEG_LENS = (7, 5)
SEQ_LEN = sum(SEG_LENS)
TOP_K = 16


def _tiny_model_cfg() -> Qwen3DenseConfig:
    # attn_impl pinned to the eager op: the flash path needs flash-attn 2 (absent here) while the
    # installed flash-attn-3 dist suppresses MHAConfig's flex fallback but stays unused unless
    # XTUNER_USE_FA3=1, so the default "flash_attention" crashes at runtime. Eager keeps these
    # trainer tests attention-backend-agnostic (HF-bitwise path, no torch.compile).
    return Qwen3DenseConfig(
        vocab_size=VOCAB,
        max_position_embeddings=512,
        eos_token_id=0,
        bos_token_id=1,
        pad_token_id=2,
        num_hidden_layers=LAYERS,
        hidden_size=HIDDEN,
        intermediate_size=64,  # multiple of the w4a8 per-group group_size (64)
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention=MHAConfig(
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, qk_norm=True, attn_impl="eager_attention"
        ),
        compile_cfg=False,
        tie_word_embeddings=False,
        max_window_layers=LAYERS,
    )


def _quant_cfg(schema: str, kv: KVQConfig | None, freeze_ratio: float = 0.0) -> QADQuantConfig:
    # "lm_head" must stay bf16: the design default (§4.1) and its two-argument
    # (hidden_states, loss_ctx) forward contract both require it.
    selective = SelectiveQuantConfig(exclude=["lm_head"])
    return QADQuantConfig(
        quant_schema=schema,
        w4_granularity="per_group",
        group_size=64,
        selective=selective,
        kv_cache=kv,
        kv_amax_ema_decay=0.9,
        scale_freeze_last_ratio=freeze_ratio,
    )


def _make_batch(seed: int = 0) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (1, SEQ_LEN), generator=generator, device="cpu")
    cu_seq_lens = torch.tensor([0, SEG_LENS[0], SEQ_LEN], dtype=torch.int32, device="cpu")
    shifted_labels = input_ids.roll(-1, dims=1).clone()
    seq_ctx = SequenceContext(
        input_ids=input_ids,
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens,
        max_length_q=max(SEG_LENS),
        max_length_k=max(SEG_LENS),
        num_padding=0,
        device="cpu",
    )
    return {"seq_ctx": seq_ctx, "shifted_labels": shifted_labels}


def _build_plain_teacher() -> nn.Module:
    model = _tiny_model_cfg().build()
    model.init_weights()
    model.eval()
    model.to(device=DEVICE)
    return model


def _save_teacher_ckpt(hf_dir: Path) -> Path:
    model = _tiny_model_cfg().build()
    model.init_weights()
    model.save_hf(hf_dir=hf_dir)
    return hf_dir


def _teacher_adapter_cfg(model_path: Path | None) -> QADTeacherAdapterConfig:
    return QADTeacherAdapterConfig(
        teacher=TrainTeacherConfig(
            name="teacher",
            model_path=str(model_path) if model_path is not None else "stub",
            model_cfg=_tiny_model_cfg(),
            fsdp_cfg=FSDPConfig(tp_size=1, recompute_ratio=0, torch_compile=False, requires_grad=False),
        ),
        teacher_offload=False,
        chunk_size=4,  # smaller than the segment length so chunk boundaries are exercised
    )


class _FixedLoader:
    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def get_state_dict(self) -> dict[str, Any]:
        # Checkpoint-compatible no-op state: the fixed loader replays a pre-built list, so there
        # is no consumption cursor to persist.
        return {"num_batches": len(self.batches)}


class _FixedBatchDataloaderConfig(BaseDataloaderConfig):
    batches: list = []
    pad_token_id: int | None = None

    def build(self, **kwargs) -> _FixedLoader:
        return _FixedLoader(self.batches)


class _RecordingQADTrainer(QADTrainer):
    """Records the tensor fields attached to the batch once the seam has
    populated them."""

    def __init__(self, **kwargs) -> None:
        self.recorded: list[dict[str, tuple[int, ...]]] = []
        super().__init__(**kwargs)

    def _prepare_model_input(self, data_batch):
        result = super()._prepare_model_input(data_batch)
        self.recorded = [
            {name: tuple(value.shape) for name, value in item.items() if isinstance(value, torch.Tensor)}
            for item in data_batch
        ]
        return result


class _StubSelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_key_value_heads = 2
        self.k_proj = nn.Linear(HIDDEN, 4, bias=False)
        self.v_proj = nn.Linear(HIDDEN, 4, bias=False)


class _StubBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _StubSelfAttn()


def _grad_abs_sum(grad: torch.Tensor) -> float:
    full = grad.full_tensor() if isinstance(grad, DTensor) else grad
    return float(full.abs().sum().item())


def _param_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU clone of every master weight; DTensors are gathered to full tensors
    first."""
    snapshot: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        full = param.full_tensor() if isinstance(param, DTensor) else param
        snapshot[name] = full.detach().cpu().clone()
    return snapshot


def _minimal_config_dict() -> dict[str, Any]:
    return {
        "trainer": TrainerConfig(
            model_cfg=_tiny_model_cfg(),
            optim_cfg=AdamWConfig(lr=1e-4),
            dataloader_cfg=_FixedBatchDataloaderConfig(batches=[]),
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0.0, lr_min=1e-4),
            global_batch_size=1,
            total_step=1,
        ),
        "quant": _quant_cfg("w8a8", KVQConfig(dtype="int8")),
        "teacher": _teacher_adapter_cfg(None),
    }


class TestQADTrainerConfig:
    """Config-level validation, pure CPU."""

    def _minimal(self, **overrides: Any) -> QADTrainerConfig:
        fields = _minimal_config_dict()
        fields.update(overrides)
        return QADTrainerConfig(**fields)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._minimal(bogus_field=1)

    def test_lora_trainable_rejected_at_resolution(self) -> None:
        config = self._minimal(trainable="lora")
        with pytest.raises(NotImplementedError, match="lora"):
            resolve_qad_trainer_config(config)

    def test_holdout_interval_validated(self) -> None:
        with pytest.raises(ValidationError):
            HoldoutConfig(interval=0)

    def test_loss_mode_passthrough_reaches_loss_context(self) -> None:
        loss_cfg = QADLossConfig(mode="qad_full_kl", top_k=8)
        config = self._minimal(loss=loss_cfg)
        resolved = resolve_qad_trainer_config(config)
        assert resolved.loss_cfg is loss_cfg
        data = {"loss_mask": torch.ones(1, 4), "teacher_logprobs": torch.zeros(1, 4, VOCAB)}
        ctx = loss_cfg.build(data)
        assert ctx is not None
        assert ctx.loss_cfg is loss_cfg
        assert ctx.loss_cfg.mode == "qad_full_kl"

    def test_resolve_overrides_load_from_and_hub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XTUNER_FROM_HUB", raising=False)
        config = self._minimal(model_path="ms://some/model")
        resolved = resolve_qad_trainer_config(config)
        assert resolved.load_from == "ms://some/model"
        assert os.environ.get("XTUNER_FROM_HUB") == "modelscope"


class TestQADTeacherTargetSeam:
    """The trainer's teacher-target seam, pure CPU, no FSDP."""

    def _bare_trainer(self, adapter: QADTeacherAdapter, teacher: nn.Module) -> QADTrainer:
        trainer = QADTrainer.__new__(QADTrainer)
        trainer._qad_config = None
        trainer._teacher_adapter = adapter
        trainer._teacher_model = _QADTeacherForward(teacher)
        return trainer

    def test_teacher_forward_wrapper_shape(self) -> None:
        teacher = _build_plain_teacher()
        batch = _make_batch()
        wrapper = _QADTeacherForward(teacher)
        seq_ctx = batch["seq_ctx"]
        logits = wrapper(input_ids=seq_ctx.input_ids.to(DEVICE), position_ids=seq_ctx.position_ids.to(DEVICE))
        assert logits.shape == (1, SEQ_LEN, VOCAB)

    def test_topk_targets_contract_and_packing(self) -> None:
        teacher = _build_plain_teacher()
        adapter = QADTeacherAdapter(_teacher_adapter_cfg(None), VOCAB)
        trainer = self._bare_trainer(adapter, teacher)
        batch = _make_batch()
        loss_cfg = QADLossConfig(mode="qad_topk_forward", top_k=TOP_K)

        targets = trainer._teacher_targets(batch["seq_ctx"], loss_cfg)
        assert targets["teacher_logprobs"].shape == (1, SEQ_LEN, TOP_K)
        assert targets["target_token_ids"].shape == (1, SEQ_LEN, TOP_K)
        assert targets["target_token_ids"].dtype == torch.int64

        # The packed row must be split at the cu_seq_lens boundaries: segment targets equal a
        # teacher forward over the segment alone.
        first_end = SEG_LENS[0]
        seg_ids = batch["seq_ctx"].input_ids[:, :first_end].to(DEVICE)
        seg_pos = batch["seq_ctx"].position_ids[:, :first_end].to(DEVICE)
        seg_targets = adapter.compute_targets(
            _QADTeacherForward(teacher), {"input_ids": seg_ids, "position_ids": seg_pos}, loss_cfg
        )
        assert torch.allclose(
            targets["teacher_logprobs"][0, :first_end], seg_targets["teacher_logprobs"][0], atol=1e-6
        )
        assert torch.equal(targets["target_token_ids"][0, :first_end], seg_targets["target_token_ids"][0])

    def test_full_kl_targets_contract(self) -> None:
        teacher = _build_plain_teacher()
        adapter = QADTeacherAdapter(_teacher_adapter_cfg(None), VOCAB)
        trainer = self._bare_trainer(adapter, teacher)
        batch = _make_batch()
        loss_cfg = QADLossConfig(mode="qad_full_kl")
        targets = trainer._teacher_targets(batch["seq_ctx"], loss_cfg)
        assert targets["teacher_logprobs"].shape == (1, SEQ_LEN, VOCAB)
        assert "target_token_ids" not in targets

    def test_loss_mask_ignores_padding(self) -> None:
        loss_cfg = QADLossConfig()
        batch = _make_batch()
        labels = batch["shifted_labels"].clone()
        labels[0, :3] = -100
        mask = _loss_mask({"shifted_labels": labels}, loss_cfg)
        assert mask.shape == labels.shape
        assert float(mask[0, :3].sum()) == 0.0
        assert float(mask[0, 3:].sum()) == SEQ_LEN - 3


class TestQADScalePhase:
    """The step-counter freeze arithmetic, pure CPU."""

    def _bare_trainer(
        self, total_step: int, freeze_ratio: float
    ) -> tuple[QADTrainer, list[FakeQuantLinear], list[Any]]:
        quant = _quant_cfg("w8a8", KVQConfig(dtype="int8"), freeze_ratio=freeze_ratio)
        trainer = QADTrainer.__new__(QADTrainer)
        trainer._qad_config = QADTrainerConfig(
            trainer=TrainerConfig(
                model_cfg=_tiny_model_cfg(),
                optim_cfg=AdamWConfig(lr=1e-4),
                dataloader_cfg=_FixedBatchDataloaderConfig(batches=[]),
                lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0.0, lr_min=1e-4),
                global_batch_size=1,
                total_step=total_step,
            ),
            quant=quant,
            teacher=_teacher_adapter_cfg(None),
        )
        trainer._total_step = total_step
        trainer._cur_step = 0
        trainer._scales_frozen = False
        trainer._fake_quant_modules = [FakeQuantLinear(4, 4, bias=False)]
        trainer._kv_handles = apply_kv_quantization(_StubBlock(), KVQConfig(dtype="int8"), 0.9)
        trainer.logger = logging.getLogger("qad_scale_phase_test")
        return trainer, trainer._fake_quant_modules, trainer._kv_handles

    def test_freeze_engages_at_threshold_step(self) -> None:
        trainer, modules, handles = self._bare_trainer(total_step=2, freeze_ratio=0.5)
        trainer._advance_scale_phase()  # step 0: cur_step + 1 = 1 > 1 is False
        assert all(not module._scales_frozen for module in modules)
        assert all(not handle.quantizer.frozen for handle in handles)
        trainer._cur_step = 1
        trainer._advance_scale_phase()  # step 1: cur_step + 1 = 2 > 1 is True
        assert all(module._scales_frozen for module in modules)
        assert all(handle.quantizer.frozen for handle in handles)
        assert trainer._scales_frozen

    def test_no_freeze_with_zero_ratio(self) -> None:
        trainer, modules, handles = self._bare_trainer(total_step=4, freeze_ratio=0.0)
        for step in range(4):
            trainer._cur_step = step
            trainer._advance_scale_phase()
        assert all(not module._scales_frozen for module in modules)
        assert all(not handle.quantizer.frozen for handle in handles)


def _broadcast_tmp_dir() -> Path:
    holder = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(holder, src=0)
    return Path(str(holder[0]))


class TestQADTrainerE2E(DeterministicDDPTestCase):
    """End-to-end one-step training through the real Trainer/FSDP pipeline."""

    def create_pg(self, device):
        ret = super().create_pg(device)
        os.environ["LOCAL_RANK"] = str(dist.get_rank() % max(torch.cuda.device_count(), 1))
        # DistributedTestBase sets the default tensor device to `device`; reset it to CPU so
        # repo paths that use CPU RNG generators under an ambient device (e.g. the randperm in
        # Dense.fully_shard) keep working in the spawned harness. The trainer moves every
        # tensor to the training device explicitly.
        torch.set_default_device("cpu")
        return ret

    @property
    def world_size(self) -> int:
        return 1

    def _run_one_step(
        self,
        quant_cfg: QADQuantConfig,
        work_dir: Path,
        total_step: int = 1,
        seeds: tuple[int, ...] = (0,),
        checkpoint_interval: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], QADTrainer | None]:
        records: dict[str, Any] = {
            "losses": [],
            "student_grad_nonzero": False,
            "teacher_grads_none": False,
            "quantizer_params": -1,
            "recorded": [],
            "frozen_w": [],
            "frozen_kv": [],
            "weights_before": {},
            "weights_after": {},
            "cur_step": -1,
        }
        holder: list[QADTrainer] = []

        def hook(train_step_info, step, epoch, total_step, total_epoch) -> None:
            trainer = holder[0]
            records["losses"].append(float(train_step_info["total_loss"]))
            model = trainer._engine.model
            grads = [param.grad for param in model.parameters() if param.grad is not None]
            records["student_grad_nonzero"] = any(_grad_abs_sum(grad) > 0.0 for grad in grads)
            teacher = trainer._teacher_model
            assert teacher is not None
            records["teacher_grads_none"] = all(param.grad is None for param in teacher.model.parameters())
            records["quantizer_params"] = sum(
                param.numel() for handle in trainer._kv_handles for param in handle.quantizer.parameters()
            )
            records["recorded"] = trainer.recorded
            records["frozen_w"].append(all(module._scales_frozen for module in trainer._fake_quant_modules))
            records["frozen_kv"].append(all(handle.quantizer.frozen for handle in trainer._kv_handles))

        ckpt_dir = _save_teacher_ckpt(work_dir / "teacher_ckpt")
        # One optimizer step consumes one data_batch = list of micro-batch dicts.
        batches = [[_make_batch(seed=seed)] for seed in seeds]
        trainer_cfg = TrainerConfig(
            model_cfg=_tiny_model_cfg(),
            optim_cfg=AdamWConfig(lr=1e-4, weight_decay=0.0),
            fsdp_cfg=FSDPConfig(tp_size=1, recompute_ratio=0, torch_compile=False),
            dataloader_cfg=_FixedBatchDataloaderConfig(batches=batches),
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0.0, lr_min=1e-4),
            global_batch_size=1,
            total_step=total_step,
            work_dir=work_dir / "work_dir",
            checkpoint_interval=checkpoint_interval,
            snapshot_interval=None,
            hf_max_keep=None,
            check_health_interval=None,
            profile_time=False,
            profile_memory=False,
            hooks_config=HooksConfig(after_train_step=hook),
        )
        qad_cfg = QADTrainerConfig(
            trainer=trainer_cfg,
            quant=quant_cfg,
            teacher=_teacher_adapter_cfg(ckpt_dir),
            loss=QADLossConfig(mode="qad_topk_forward", top_k=TOP_K),
        )
        trainer = _RecordingQADTrainer.from_config(qad_cfg)
        holder.append(trainer)
        records["weights_before"] = _param_snapshot(trainer._engine.model)
        trainer.fit()
        records["weights_after"] = _param_snapshot(trainer._engine.model)
        records["cur_step"] = trainer.cur_step
        state = {
            "fake_quant_count": len(trainer._fake_quant_modules),
            "kv_handle_count": len(trainer._kv_handles),
            "loss_cfg_mode": trainer._qad_loss_cfg().mode,
        }
        return records, state, trainer

    def test_one_step_three_configs(self) -> None:
        self.create_pg(DEVICE)
        tmp_dir = _broadcast_tmp_dir()
        try:
            configs = [
                ("w8a8-kvint8", _quant_cfg("w8a8", KVQConfig(dtype="int8"))),
                ("w4a8-kvint8", _quant_cfg("w4a8", KVQConfig(dtype="int8"))),
                ("w8a8-kvbf16", _quant_cfg("w8a8", None)),
            ]
            for name, quant_cfg in configs:
                records, state, trainer = self._run_one_step(quant_cfg, tmp_dir / name)
                del trainer
                assert state["fake_quant_count"] > 0
                if quant_cfg.kv_cache is not None:
                    assert state["kv_handle_count"] > 0
                else:
                    assert state["kv_handle_count"] == 0
                assert state["loss_cfg_mode"] == "qad_topk_forward"

                assert len(records["losses"]) == 1
                assert math.isfinite(records["losses"][0])
                assert records["student_grad_nonzero"]
                assert records["teacher_grads_none"]
                assert records["quantizer_params"] == 0

                # Targets contract on the attached batch items.
                assert records["recorded"], "no batch recorded"
                shapes = records["recorded"][0]
                assert shapes["teacher_logprobs"] == (1, SEQ_LEN, TOP_K)
                assert shapes["target_token_ids"] == (1, SEQ_LEN, TOP_K)
                assert shapes["loss_mask"] == (1, SEQ_LEN)
            dist.barrier()
        finally:
            if dist.get_rank() == 0:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_two_step_checkpoint_roundtrip_preserves_frozen_forward(self) -> None:
        """D-3: checkpoint save -> load round-trip restores a bit-exact frozen-
        state forward."""
        self.create_pg(DEVICE)
        tmp_dir = _broadcast_tmp_dir()
        try:
            quant_cfg = _quant_cfg("w8a8", KVQConfig(dtype="int8"), freeze_ratio=0.5)
            records, _state, trainer = self._run_one_step(
                quant_cfg, tmp_dir / "roundtrip", total_step=2, seeds=(1, 2), checkpoint_interval=1
            )
            holdout_batches = [[_make_batch(seed=3)]]
            holdout_before = trainer.evaluate_holdout(holdout_batches)

            # Perturb one weight: the frozen-state forward must react to weight changes.
            with torch.no_grad():
                first_param = next(trainer._engine.model.parameters())
                first_param.add_(0.05)
            holdout_perturbed = trainer.evaluate_holdout(holdout_batches)

            # Then the checkpoint round-trip (saved by fit at checkpoint_interval=1) must restore
            # the exact pre-perturbation behavior, including the frozen train state.
            checkpoint_path = Path(trainer.meta.latest_exp.checkpoint_list[-1])
            assert f"step-{trainer.cur_step}" in str(checkpoint_path)
            trainer._engine.load_dcp(weights_dir=checkpoint_path / trainer._SAVE_WEIGHTS_DIR, load_states=False)
            with (checkpoint_path / trainer._SAVE_TRAIN_STATE_PATH).open("r") as f:
                train_state = json.load(f)
            trainer._cur_step = train_state["cur_step"]
            holdout_after = trainer.evaluate_holdout(holdout_batches)
            del trainer

            assert records["cur_step"] == 2
            assert train_state["cur_step"] == 2
            assert holdout_perturbed != holdout_before, "frozen-state forward ignores weight changes"
            assert holdout_after == holdout_before, "checkpoint round-trip broke the frozen-state forward"
            dist.barrier()
        finally:
            if dist.get_rank() == 0:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_two_step_freeze_engages(self) -> None:
        self.create_pg(DEVICE)
        tmp_dir = _broadcast_tmp_dir()
        try:
            quant_cfg = _quant_cfg("w8a8", KVQConfig(dtype="int8"), freeze_ratio=0.5)
            records, _state, trainer = self._run_one_step(quant_cfg, tmp_dir / "freeze", total_step=2, seeds=(1, 2))
            holdout_kl = trainer.evaluate_holdout([[_make_batch(seed=3)]])
            del trainer
            assert len(records["losses"]) == 2
            assert records["frozen_w"] == [False, True]
            assert records["frozen_kv"] == [False, True]
            assert math.isfinite(holdout_kl)

            # D-1: the trainer's global step counter advanced exactly with the micro-run.
            assert records["cur_step"] == 2
            # D-2: master weights actually moved.
            before, after = records["weights_before"], records["weights_after"]
            moved = [name for name in before if not torch.equal(before[name], after[name])]
            assert moved, "no master weight moved during the 2-step micro-run"
            dist.barrier()
        finally:
            if dist.get_rank() == 0:
                shutil.rmtree(tmp_dir, ignore_errors=True)
