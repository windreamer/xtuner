# Copyright (c) OpenMMLab. All rights reserved.
"""QAD (quantization-aware distillation) SFT trainer.

Integrates the :mod:`xtuner.v1.quant` package into the SFT ``Trainer``: the student is a
fake-quant model (W8A8/W4A8 + KV-cache int8 simulation), the teacher is a frozen BF16 copy of
the same architecture, and the loss is the QAD forward KL (design doc §8).

Seam inventory — core ``xtuner/v1/train/trainer.py`` stays untouched; three overrides carry the
whole integration:

- ``build_engine``: applies ``apply_quantization`` + ``apply_kv_quantization`` to the student right
  after ``super().build_engine`` has materialized the weights. The base ``TrainEngine`` constructs
  the model on the meta device and FSDP2-wraps it inside its own constructor, so "apply before
  FSDP wrapping" (§5.4) is not expressible from a subclass; the in-place ``__class__`` swap used
  by ``apply_quantization`` keeps the wrapped module tree, parameter registration and
  ``state_dict`` keys intact, and the application still happens before any forward pass and
  before any scale caching, which is the observable part of the contract. KV handles are attached
  per attention root selected by ``_attention_modules`` — a whole-model ``apply_kv_quantization``
  call would match the quantizer submodules that ``FakeQuantLinear`` registers under attention
  module paths.
- ``_prepare_model_input``: runs once per optimizer step, before every forward of the step. It
  (a) advances the §5.3 scale lifecycle — ``FakeQuantLinear.begin_step()`` while training,
  ``freeze_scales()`` + KV ``set_frozen(True)`` once ``step > total_steps *
  (1 - scale_freeze_last_ratio)`` — and (b) computes the teacher targets for the packed batch,
  per packed segment, so the teacher's causal attention respects the same packing boundaries as
  the student's varlen attention, then attaches ``teacher_logprobs``/``target_token_ids`` and the
  ``loss_mask`` (derived from ``shifted_labels``) to each micro-batch item before delegating to
  ``Trainer._prepare_model_input``, where ``model.build_loss_ctx_batch`` consumes them through
  ``QADLossConfig.build``. The model's ``lm_loss_cfg`` must be the ``QADLossConfig`` carried by
  ``QADTrainerConfig.loss``; :func:`resolve_qad_trainer_config` forwards it through
  ``TrainerConfig.loss_cfg`` (pydantic stores sibling model instances in model-typed fields
  without revalidation, and ``Trainer``'s backward-compat wiring copies ``loss_cfg`` into
  ``model_cfg.lm_loss_cfg``).
- ``_log_step``: logs ``qad/train_kl``, ``qad/topk_overlap_fraction`` (top-K mode) and
  ``qad/teacher_time_{compute,onload,offload}`` (§7.3) through the trainer's exp tracker and
  logger.

Documented narrowings:

- ``trainable="lora"`` raises ``NotImplementedError`` at config resolution: the base engine
  builds the optimizer over the model inside ``TrainEngine.__init__``, before the trainer subclass
  could attach adapters into ``FakeQuantLinear``, so adapter-only optimizer param groups cannot be
  expressed without changing the engine. Full-parameter training is the implemented path
  (``FakeQuantLinear`` itself already supports the ``delta_w`` contract for later use).
- Holdout evaluation: the SFT ``Trainer`` has no evaluation seam in ``fit``; ``QADTrainer`` exposes
  :meth:`evaluate_holdout` (teacher-vs-student masked mean KL over packed batches, no grad) for
  recipe hooks, and :class:`HoldoutConfig` validates the interval, but interval-driven evaluation
  is not wired into the training loop.
"""

import os
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from xtuner.v1.config import FSDPConfig, OptimConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.engine import TrainEngine
from xtuner.v1.engine.train_engine import TrainStepInfo
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.base import BaseModel as XtunerBaseModel
from xtuner.v1.model.base import ModelItem, TransformerConfig, XTunerBaseModelConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.model.utils.misc import ModelForwardExtraLogInfo
from xtuner.v1.quant.config import QADQuantConfig
from xtuner.v1.quant.fake_quant_linear import FakeQuantLinear, apply_quantization
from xtuner.v1.quant.kv_cache import KVQuantHandle, apply_kv_quantization
from xtuner.v1.quant.loss import QADLossConfig
from xtuner.v1.quant.teacher import QADTeacherAdapter, QADTeacherAdapterConfig, model_vocab_size
from xtuner.v1.rl.distillation.train_teacher_manager import TeacherTiming
from xtuner.v1.rl.model_utils import build_frozen_model
from xtuner.v1.train.trainer import (
    InternalMetrics,
    PerformanceStatistics,
    Trainer,
    TrainerConfig,
)
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


class HoldoutConfig(BaseModel):
    """Holdout evaluation configuration.

    The SFT ``Trainer`` loop has no evaluation seam, so this config only validates the interval;
    run :meth:`QADTrainer.evaluate_holdout` from a training hook to produce ``qad/holdout_kl``.

    Args:
        interval (int): Evaluate holdout KL every N optimizer steps. Must be positive.
    """

    model_config = ConfigDict(extra="forbid")

    interval: int = Field(default=100, gt=0)


class QADTrainerConfig(BaseModel):
    """Configuration of the QAD trainer (design doc §8.1).

    Args:
        trainer (TrainerConfig): The base SFT trainer config (model, optimizer, data, FSDP,
            checkpointing), reused as-is.
        quant (QADQuantConfig): Quantization schema, layer selection, KV-cache config and the
            scale-freeze ratio.
        teacher (QADTeacherAdapterConfig): Frozen BF16 teacher config
            (``TrainTeacherConfig`` + ``teacher_offload`` + ``chunk_size``).
        loss (QADLossConfig): QAD loss configuration; forwarded to the student model as its
            ``lm_loss_cfg``. Defaults to the top-K forward-KL mode.
        model_path (str | None): Student model path or ModelScope repo id overriding
            ``trainer.load_from``; ``None`` keeps ``trainer.load_from``.
        from_hub (Literal["huggingface", "modelscope"]): Hub used to resolve ``model_path``
            repo ids; ``"modelscope"`` also sets the ``XTUNER_FROM_HUB`` env fallback (design
            doc §1.4/§5.7). Defaults to ``"modelscope"``.
        trainable (Literal["full", "lora"]): ``"full"`` trains the master weights; ``"lora"`` is
            validated but not implemented and raises ``NotImplementedError`` at config
            resolution.
        holdout (HoldoutConfig | None): Optional holdout evaluation interval; see
            :class:`HoldoutConfig` for the current scope.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, protected_namespaces=())

    trainer: TrainerConfig
    quant: QADQuantConfig
    teacher: QADTeacherAdapterConfig
    loss: QADLossConfig = Field(default_factory=QADLossConfig)
    model_path: str | None = None
    from_hub: Literal["huggingface", "modelscope"] = "modelscope"
    trainable: Literal["full", "lora"] = "full"
    holdout: HoldoutConfig | None = None


def resolve_qad_trainer_config(config: QADTrainerConfig) -> TrainerConfig:
    """Materialize the base ``TrainerConfig`` consumed by
    ``Trainer.from_config``.

    Validates the QAD-only fields and forwards ``config.loss`` into ``loss_cfg`` so that
    ``Trainer``'s backward-compat wiring installs it as the student model's ``lm_loss_cfg``.

    Args:
        config (QADTrainerConfig): The QAD training configuration.

    Returns:
        TrainerConfig: A deep copy of ``config.trainer`` with ``loss_cfg`` set to the QAD loss
        config and ``load_from`` overridden by ``model_path`` when given.
    """
    if config.trainable == "lora":
        raise NotImplementedError(
            'QAD trainable="lora" is not implemented: the base engine builds the optimizer over the '
            "model inside TrainEngine.__init__, before QADTrainer could attach adapters into "
            'FakeQuantLinear; use trainable="full" (the default).'
        )
    trainer_cfg = config.trainer.model_copy(deep=True)
    if config.model_path is not None:
        trainer_cfg.load_from = config.model_path
    # pydantic keeps model instances of model-typed fields as-is (no revalidation), so the
    # CELossConfig-annotated field carries the QADLossConfig instance through to the trainer.
    trainer_cfg.loss_cfg = cast(CELossConfig, config.loss)
    if config.from_hub == "modelscope":
        # Config-explicit hub choice (design doc §5.7); a pre-set env value still wins.
        os.environ.setdefault("XTUNER_FROM_HUB", config.from_hub)
    return trainer_cfg


class _QADTeacherForward(nn.Module):
    """Adapter-facing callable around a frozen xtuner teacher model.

    ``QADTeacherAdapter.compute_targets`` calls its teacher as ``model(input_ids=..., position_ids=...,
    attention_mask=...)``, while xtuner models take a ``SequenceContext``. The wrapper translates the plain-call
    contract into a single-segment packed ``SequenceContext`` (packing boundaries are handled by the trainer, which
    invokes the teacher once per packed segment) and unwraps ``ModelOutputs.logits``. ``to_device`` mirrors the xtuner
    model convention consumed by the adapter's on/offload path.
    """

    def __init__(self, model: XtunerBaseModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask  # packing is expressed through cu_seq_lens; this call is one segment
        seq_len = int(input_ids.shape[1])
        cu_seq_lens = cast(torch.IntTensor, torch.IntTensor([0, seq_len]).to(input_ids.device))
        seq_ctx = SequenceContext(
            input_ids=input_ids,
            cu_seq_lens_q=cu_seq_lens,
            cu_seq_lens_k=cu_seq_lens,
            max_length_q=seq_len,
            max_length_k=seq_len,
            num_padding=0,
            device=str(input_ids.device),
            position_ids=position_ids,
        )
        output = self.model(seq_ctx=seq_ctx, loss_ctx=None)
        logits = output.logits
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"the QAD teacher forward returned no logits tensor, got {type(logits)!r}")
        return logits

    def to_device(self, device: str) -> None:
        to_device = getattr(self.model, "to_device", None)
        if callable(to_device):
            to_device(device)
        else:
            self.model.to(device=device)


def _trainer_init_kwargs(config: TrainerConfig) -> dict[str, Any]:
    """Map a resolved :class:`TrainerConfig` onto :class:`Trainer.__init__`
    keyword arguments.

    Mirrors ``Trainer.from_config`` field-for-field so that ``QADTrainer.from_config`` can pass
    ``qad_config`` into the constructor (the quantization hooks run inside ``build_engine``, which
    the base constructor invokes).

    Args:
        config (TrainerConfig): The resolved trainer configuration.

    Returns:
        dict[str, Any]: Keyword arguments for ``Trainer.__init__``.
    """
    return {
        "load_from": config.load_from,
        "model_cfg": config.model_cfg,
        "optim_cfg": config.optim_cfg,
        "fsdp_cfg": config.fsdp_cfg,
        "dataset_cfg": config.dataset_cfg,
        "dataloader_cfg": config.dataloader_cfg,
        "loss_cfg": config.loss_cfg,
        "lr_cfg": config.lr_cfg,
        "tokenizer_path": config.tokenizer_path,
        "global_batch_size": config.global_batch_size,
        "work_dir": config.work_dir,
        "log_dir": config.log_dir,
        "sp_size": config.sp_size,
        "total_step": config.total_step,
        "total_epoch": config.total_epoch,
        "resume_cfg": config.resume_cfg,
        "auto_resume": config.auto_resume,
        "load_checkpoint_cfg": config.load_checkpoint_cfg,
        "strict_load": config.strict_load,
        "checkpoint_interval": config.checkpoint_interval,
        "checkpoint_maxkeep": config.checkpoint_maxkeep,
        "async_hf_export": config.async_hf_export,
        "skip_checkpoint_validation": config.skip_checkpoint_validation,
        "patch_for_dcp_finish": config.patch_for_dcp_finish,
        "async_checkpoint": config.async_checkpoint,
        "snapshot_interval": config.snapshot_interval,
        "check_health_interval": config.check_health_interval,
        "hf_interval": config.hf_interval,
        "hf_max_keep": config.hf_max_keep,
        "exp_tracker": config.exp_tracker,
        "profile_step": config.profile_step,
        "profile_time_step": config.profile_time_step,
        "profile_memory_step": config.profile_memory_step,
        "profile_time": config.profile_time,
        "profile_memory": config.profile_memory,
        "intra_layer_micro_batch": config.intra_layer_micro_batch,
        "seed": config.seed,
        "backend": config.dist_backend,
        "debug": config.debug,
        "debug_skip_save": config.debug_skip_save,
        "prober_list": config.prober_list,
        "do_clip": config.do_clip,
        "grad_norm_dtype": config.grad_norm_dtype,
        "hooks_config": config.hooks_config,
        "trainer_cfg": config,
        "internal_metrics_cfg": config.internal_metrics_cfg,
    }


class QADTrainer(Trainer):
    """SFT trainer that trains a fake-quant student against a frozen BF16
    teacher.

    Args:
        qad_config (QADTrainerConfig | None): The QAD configuration; ``None`` is only accepted so
            that the base ``Trainer`` construction path stays reachable — every QAD usage must
            build via :meth:`from_config`.
        **trainer_kwargs (Any): Keyword arguments of :class:`~xtuner.v1.train.trainer.Trainer`;
            ``loss_cfg`` must resolve to the QAD loss config (``from_config`` guarantees this).
    """

    def __init__(self, *, qad_config: QADTrainerConfig | None = None, **trainer_kwargs: Any) -> None:
        # Assigned before super().__init__ because the base constructor calls build_engine.
        self._qad_config = qad_config
        self._kv_handles: list[KVQuantHandle] = []
        self._fake_quant_modules: list[FakeQuantLinear] = []
        self._teacher_adapter: QADTeacherAdapter | None = None
        self._teacher_model: _QADTeacherForward | None = None
        self._scales_frozen = False
        self._last_teacher_timings: TeacherTiming | None = None
        super().__init__(**trainer_kwargs)
        if qad_config is not None:
            # Fail fast when the loss passthrough did not reach the model config.
            self._qad_loss_cfg()

    @classmethod
    def from_config(cls, config: QADTrainerConfig) -> "QADTrainer":  # type: ignore[override]
        """Create a :class:`QADTrainer` from a :class:`QADTrainerConfig`.

        The QAD config schema replaces the base ``TrainerConfig`` argument on purpose; the
        resolved base config is materialized by :func:`resolve_qad_trainer_config`.

        Args:
            config (QADTrainerConfig): The QAD training configuration.

        Returns:
            QADTrainer: The trainer instance, retaining ``config`` for the quant/teacher hooks.
        """
        trainer_cfg = resolve_qad_trainer_config(config)
        self = cls(qad_config=config, **_trainer_init_kwargs(trainer_cfg))
        self._config = trainer_cfg
        self._print_training_config()
        return self

    def build_engine(
        self,
        model_path: Path | None,
        model_config: XTunerBaseModelConfig,
        optim_config: OptimConfig,
        fsdp_config: FSDPConfig,
        load_checkpoint_path: str | Path | None,
        intra_layer_micro_batch: int = 1,
        strict: bool = True,
    ) -> TrainEngine:
        """Build the base engine, then apply fake-quant and KV simulation to
        the student.

        The quantization state carries no parameters, so applying it after the engine (and its
        optimizer) are built changes neither the optimized parameter set nor the ``state_dict``
        layout.

        Args:
            model_path (Any): Path to the model checkpoint, or ``None`` for fresh init.
            model_config (XTunerBaseModelConfig): The student model configuration.
            optim_config (OptimConfig): The optimizer configuration.
            fsdp_config (FSDPConfig | None): The FSDP configuration.
            load_checkpoint_path (Any): Trainer checkpoint path to resume from, or ``None``.
            intra_layer_micro_batch (int): Intra-layer micro batch size. Defaults to 1.
            strict (bool): Whether to strictly load model weights. Defaults to True.

        Returns:
            TrainEngine: The engine with ``FakeQuantLinear`` modules and KV quantization handles
            installed.
        """
        engine = super().build_engine(
            model_path=model_path,
            model_config=model_config,
            optim_config=optim_config,
            fsdp_config=fsdp_config,
            load_checkpoint_path=load_checkpoint_path,
            intra_layer_micro_batch=intra_layer_micro_batch,
            strict=strict,
        )
        quant_cfg = self._require_qad_config().quant
        apply_quantization(engine.model, quant_cfg)
        self._fake_quant_modules = [module for module in engine.model.modules() if isinstance(module, FakeQuantLinear)]
        self._kv_handles = []
        if quant_cfg.kv_cache is not None:
            for _name, attention in _attention_modules(engine.model):
                self._kv_handles.extend(
                    apply_kv_quantization(attention, quant_cfg.kv_cache, quant_cfg.kv_amax_ema_decay)
                )
        return engine

    def evaluate_holdout(self, data_batches: Sequence[list[dict[str, Any]]]) -> float:
        """Holdout KL of the current student against the teacher over packed
        batches.

        Replays the training seam (teacher targets -> loss context -> forward) under ``no_grad``
        and returns the masked mean per-token KL (§7.3 ``qad/holdout_kl``). The SFT ``fit`` loop
        has no evaluation seam, so recipes call this from a training hook at
        ``QADTrainerConfig.holdout.interval``.

        Args:
            data_batches (Sequence[list[dict[str, Any]]]): Holdout batches in the packed SFT
                schema (``{"seq_ctx", "shifted_labels"}`` items).

        Returns:
            float: The masked mean KL over all valid tokens; ``nan`` when a batch has none.
        """
        loss_cfg = self._qad_loss_cfg()
        kl_sum = 0.0
        token_sum = 0.0
        with torch.no_grad():
            for data_batch in data_batches:
                for item in data_batch:
                    targets = self._teacher_targets(item["seq_ctx"], loss_cfg)
                    item["teacher_logprobs"] = targets["teacher_logprobs"]
                    if "target_token_ids" in targets:
                        item["target_token_ids"] = targets["target_token_ids"]
                    item["loss_mask"] = _loss_mask(item, loss_cfg)
                loss_ctx_dict_list = self._engine.model.build_loss_ctx_batch(data_batch, sp_mesh=self.sp_mesh)
                for item, loss_ctx_dict in zip(data_batch, loss_ctx_dict_list):
                    output = self._engine.forward_only(item["seq_ctx"].to(DEVICE), loss_ctx_dict["lm"])
                    # The loss context returns its metrics as a plain dict (ModelOutputs.extra_info).
                    extra_info = getattr(output, "extra_info", None)
                    if not isinstance(extra_info, dict):
                        continue
                    kl = extra_info["reduced_qad_kl_sum"] if "reduced_qad_kl_sum" in extra_info else None
                    count = extra_info["reduced_qad_valid_count"] if "reduced_qad_valid_count" in extra_info else None
                    if isinstance(kl, torch.Tensor) and isinstance(count, torch.Tensor):
                        kl_sum += float(kl.sum().item())
                        token_sum += float(count.sum().item())
        if token_sum == 0.0:
            return float("nan")
        return kl_sum / token_sum

    def _require_qad_config(self) -> QADTrainerConfig:
        if self._qad_config is None:
            raise ValueError("QADTrainer requires qad_config; construct it via QADTrainer.from_config")
        return self._qad_config

    def _qad_loss_cfg(self) -> QADLossConfig:
        model_cfg = self._engine.model.config
        if isinstance(model_cfg, BaseComposeConfig):
            lm_loss_cfg = model_cfg.text_config.lm_loss_cfg
        else:
            lm_loss_cfg = model_cfg.lm_loss_cfg
        if not isinstance(lm_loss_cfg, QADLossConfig):
            raise ValueError(
                "QADTrainer requires the student model's lm_loss_cfg to be a QADLossConfig, got "
                f"{type(lm_loss_cfg).__name__}; build the trainer via QADTrainer.from_config, which "
                "forwards QADTrainerConfig.loss through TrainerConfig.loss_cfg."
            )
        return lm_loss_cfg

    def _ensure_teacher(self) -> tuple[QADTeacherAdapter, _QADTeacherForward]:
        if self._teacher_adapter is None or self._teacher_model is None:
            adapter_cfg = self._require_qad_config().teacher
            teacher_cfg = adapter_cfg.teacher
            model = build_frozen_model(teacher_cfg.model_cfg, teacher_cfg.model_path, teacher_cfg.fsdp_cfg)
            student_cfg = self._engine.model.config
            if not isinstance(student_cfg, (TransformerConfig, BaseComposeConfig)):
                raise ValueError(
                    "QADTrainer expects a TransformerConfig or BaseComposeConfig student model, got "
                    f"{type(student_cfg).__name__}"
                )
            adapter = QADTeacherAdapter(adapter_cfg, model_vocab_size(student_cfg))
            teacher = _QADTeacherForward(model)
            if not adapter_cfg.teacher_offload:
                # Keep the teacher resident on the training device; the offload path moves the
                # model around every compute_targets call instead.
                teacher.to_device(str(DEVICE))
            self._teacher_model = teacher
            self._teacher_adapter = adapter
        return self._teacher_adapter, cast("_QADTeacherForward", self._teacher_model)

    def _teacher_targets(self, seq_ctx: SequenceContext, loss_cfg: QADLossConfig) -> dict[str, torch.Tensor]:
        adapter, teacher = self._ensure_teacher()
        input_ids = seq_ctx.input_ids
        if input_ids is None:
            raise ValueError("QAD teacher targets require seq_ctx.input_ids")
        seq_len = int(input_ids.shape[1])
        is_topk = loss_cfg.mode == "qad_topk_forward"
        support = loss_cfg.top_k if is_topk else adapter.teacher_vocab_size
        logprobs = torch.zeros((1, seq_len, support), dtype=torch.float32, device=DEVICE)
        token_ids = torch.zeros((1, seq_len, loss_cfg.top_k), dtype=torch.int64, device=DEVICE) if is_topk else None
        for start, end in _segment_boundaries(seq_ctx.cu_seq_lens_k):
            if end <= start:
                continue
            seg_targets = adapter.compute_targets(
                teacher, _teacher_batch(input_ids, seq_ctx.position_ids, start, end), loss_cfg
            )
            logprobs[:, start:end] = seg_targets["teacher_logprobs"].to(logprobs.device)
            if token_ids is not None:
                ids = seg_targets.get("target_token_ids")
                if not isinstance(ids, torch.Tensor):
                    raise ValueError('mode="qad_topk_forward" requires target_token_ids from the teacher adapter')
                token_ids[:, start:end] = ids.to(token_ids.device)
        merged: dict[str, torch.Tensor] = {"teacher_logprobs": logprobs}
        if token_ids is not None:
            merged["target_token_ids"] = token_ids
        return merged

    def _advance_scale_phase(self) -> None:
        quant_cfg = self._require_qad_config().quant
        freeze_from = self.total_step * (1.0 - quant_cfg.scale_freeze_last_ratio)
        if (self._cur_step + 1) > freeze_from:
            if not self._scales_frozen:
                for module in self._fake_quant_modules:
                    module.freeze_scales()
                for handle in self._kv_handles:
                    handle.quantizer.set_frozen(True)
                self._scales_frozen = True
                if self.rank == 0:
                    self.logger.info(
                        f"QAD scale freeze phase engaged at step {self._cur_step} "
                        f"(total_step={self.total_step}, freeze_from={freeze_from})"
                    )
            return
        for module in self._fake_quant_modules:
            module.begin_step()

    def _prepare_model_input(self, data_batch: list[dict[str, Any]]) -> list[ModelItem]:
        """Attach teacher targets and the loss mask to the batch, then delegate
        to the base.

        Runs once per optimizer step (design doc §8.3): the scale phase advances first, then the
        teacher targets are computed per packed segment and attached to each micro-batch item so
        that ``model.build_loss_ctx_batch`` -> ``QADLossConfig.build`` consumes them unchanged.

        Args:
            data_batch (list[dict[str, Any]]): The packed SFT micro-batches of this step.

        Returns:
            list[ModelItem]: The engine inputs built by the base trainer.
        """
        self._require_qad_config()
        self._advance_scale_phase()
        loss_cfg = self._qad_loss_cfg()
        for item in data_batch:
            targets = self._teacher_targets(item["seq_ctx"], loss_cfg)
            item["teacher_logprobs"] = targets["teacher_logprobs"]
            if "target_token_ids" in targets:
                item["target_token_ids"] = targets["target_token_ids"]
            item["loss_mask"] = _loss_mask(item, loss_cfg)
        if self._teacher_adapter is not None:
            self._last_teacher_timings = self._teacher_adapter.last_timings
        return super()._prepare_model_input(data_batch)

    def _log_step(
        self,
        train_step_info: TrainStepInfo,
        training_metrics: PerformanceStatistics,
        grad_norm: float,
        data_time: float,
        step_time: float,
        internal_metrics: InternalMetrics | None = None,
    ) -> None:
        """Log the base step info plus the §7.3 QAD metrics.

        Args:
            train_step_info (TrainStepInfo): Info returned by the engine train step; its
                ``extra_info`` carries the per-micro-batch QAD KL accumulators.
            training_metrics (PerformanceStatistics): The trainer throughput statistics.
            grad_norm (float): The clipped gradient norm of this step.
            data_time (float): Seconds spent fetching the batch.
            step_time (float): Seconds spent in the training step.
            internal_metrics (InternalMetrics | None): Model-internal metrics, if any.
        """
        super()._log_step(train_step_info, training_metrics, grad_norm, data_time, step_time, internal_metrics)
        extra_info = train_step_info.get("extra_info")
        scalars: dict[str, float] = {}
        if isinstance(extra_info, ModelForwardExtraLogInfo):
            kl_sum = extra_info["reduced_qad_kl_sum"] if "reduced_qad_kl_sum" in extra_info else None
            valid_count = extra_info["reduced_qad_valid_count"] if "reduced_qad_valid_count" in extra_info else None
            if isinstance(kl_sum, torch.Tensor) and isinstance(valid_count, torch.Tensor):
                denominator = max(float(valid_count.sum().item()), 1.0)
                scalars["qad/train_kl"] = float(kl_sum.sum().item()) / denominator
                overlap_sum = (
                    extra_info["reduced_qad_topk_overlap_fraction_sum"]
                    if "reduced_qad_topk_overlap_fraction_sum" in extra_info
                    else None
                )
                if isinstance(overlap_sum, torch.Tensor):
                    scalars["qad/topk_overlap_fraction"] = float(overlap_sum.sum().item()) / denominator
        timings = self._last_teacher_timings
        if timings is not None:
            scalars["qad/teacher_time_compute"] = timings.compute
            scalars["qad/teacher_time_onload"] = timings.onload
            scalars["qad/teacher_time_offload"] = timings.offload
        if scalars:
            self._exp_tracker.add_scalars(tag_scalar_dict=scalars, global_step=self.cur_step)
            self.logger.info("QAD metrics: " + ", ".join(f"{name}: {value:.6f}" for name, value in scalars.items()))


def _attention_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Select the attention modules of ``model`` for KV-quantization
    attachment.

    Mirrors the documented ``apply_kv_quantization`` selection (class name contains
    ``"Attention"``/``"MLA"``, or the module path contains ``"self_attn"``/``"attention"``), but
    excludes everything at or below a ``FakeQuantLinear``: the fake-quant layer registers its
    weight/activation quantizers as submodules, so paths like ``layers.0.self_attn.k_proj.
    _act_quantizer`` would otherwise match the name pattern and be mistaken for attention
    modules. The trainer therefore selects the attention roots here and calls
    ``apply_kv_quantization`` once per root, where the relative-name walk cannot hit quantizer
    paths.

    Args:
        model (nn.Module): The quantized student model.

    Returns:
        list[tuple[str, nn.Module]]: The ``(name, module)`` pairs of the selected attention roots.
    """
    selected: list[tuple[str, nn.Module]] = []
    fake_quant_prefixes: list[str] = []
    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in fake_quant_prefixes):
            continue
        if isinstance(module, FakeQuantLinear):
            fake_quant_prefixes.append(name)
            continue
        cls_name = type(module).__name__
        lowered = name.lower()
        if "Attention" in cls_name or "MLA" in cls_name or "self_attn" in lowered or "attention" in lowered:
            selected.append((name, module))
    return selected


def _segment_boundaries(cu_seq_lens: torch.Tensor) -> list[tuple[int, int]]:
    starts = cu_seq_lens.detach().to("cpu").tolist()
    return [(int(start), int(end)) for start, end in zip(starts[:-1], starts[1:])]


def _teacher_batch(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor | None,
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {"input_ids": input_ids[:, start:end].to(DEVICE)}
    if position_ids is not None:
        batch["position_ids"] = position_ids[:, start:end].to(DEVICE)
    return batch


def _loss_mask(item: dict[str, Any], loss_cfg: QADLossConfig) -> torch.Tensor:
    shifted_labels = item.get("shifted_labels")
    if not isinstance(shifted_labels, torch.Tensor):
        raise ValueError("QAD training requires shifted_labels [B, T] in the packed batch to derive the loss mask")
    return shifted_labels.ne(loss_cfg.ignore_idx).to(torch.float32)
