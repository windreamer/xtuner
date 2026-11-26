# Copyright (c) OpenMMLab. All rights reserved.
"""QAD teacher adapter: distillation targets for packed SFT batches (design doc
§7.1/§7.2).

The adapter wraps a frozen BF16 teacher (a copy of the student architecture) owned by the caller and
produces the teacher targets required by :class:`~xtuner.v1.quant.loss.QADLossConfig`:

- ``qad_topk_forward`` → ``target_token_ids [B, T, K]`` (int64) + ``teacher_logprobs [B, T, K]`` (float32);
- ``qad_full_kl`` → ``teacher_logprobs [B, T, V]`` (float32), assembled from per-chunk
  ``[B, chunk_size, V]`` log-probabilities so the full ``[B, T, V]`` teacher logits tensor is never
  materialized (§12 R5).

Temperature contract (§7.2): teacher logits are divided by ``loss_cfg.temperature`` BEFORE top-K
selection and log-softmax. Tempering preserves ordering, so the selected ids are temperature
invariant while the target log-probabilities are tempered.

Reuse vs. adaptation of ``xtuner.v1.rl.distillation`` (OPD ``TrainTeacherManager``):

- Reused: ``TrainTeacherConfig`` construction fields (``model_path``, ``model_cfg``, ``fsdp_cfg``),
  the onload → forward → offload CPU-offload cadence (``teacher_offload=True`` maps onto it), the
  ``TeacherTiming`` lifecycle accounting, and the OPD student/teacher ``vocab_size`` validation
  (§12 R7).
- Adapted for a plain training worker (this is the documented delta): the OPD manager is entangled
  with the RL-worker lifecycle — ray worker scheduling, rollout state, ``SequenceContext`` inputs,
  per-data-source ``teacher_indices`` routing, multi-teacher FSDP-collective ordering, and
  model-internal log-probability contexts (which carry no temperature). The QAD adapter drops all of
  that for a synchronous single-process path: the teacher is a plain ``torch.nn.Module`` called as
  ``model(input_ids=..., position_ids=..., attention_mask=...)`` (the subset its forward accepts)
  returning logits (tensor or object with ``.logits``). Teacher construction is caller-owned (e.g. a
  BF16 copy of the student); the adapter never instantiates models.
- The chunked teacher forward assumes a causal teacher: logits at position ``t`` depend only on
  tokens ``<= t``, so a sequence slice reproduces the full-context logits at the same positions.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import cast

import torch
from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.quant.loss import QADLossConfig
from xtuner.v1.rl.distillation.config import TrainTeacherConfig
from xtuner.v1.rl.distillation.train_teacher_manager import TeacherTiming
from xtuner.v1.utils import get_device, get_torch_device_module


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

FORWARD_KEYS: tuple[str, ...] = ("input_ids", "position_ids", "attention_mask")


def model_vocab_size(model_cfg: TransformerConfig | BaseComposeConfig) -> int:
    """Return the language-model vocabulary size of an xtuner model config.

    Args:
        model_cfg (TransformerConfig | BaseComposeConfig): The xtuner model configuration.

    Returns:
        int: The language-model ``vocab_size``; the ``text_config`` value for composed models.
    """
    if isinstance(model_cfg, BaseComposeConfig):
        return cast(TransformerConfig, model_cfg.text_config).vocab_size
    return model_cfg.vocab_size


class QADTeacherAdapterConfig(BaseModel):
    """Configuration of the QAD teacher adapter.

    Args:
        teacher (TrainTeacherConfig): The OPD training-teacher config supplying ``model_path``,
            ``model_cfg`` and ``fsdp_cfg`` of the frozen BF16 teacher (design doc §7.1).
        teacher_offload (bool): CPU-offload the teacher around every ``compute_targets`` call
            (onload → forward → offload, the OPD cadence). Saves device memory at a throughput
            cost. Defaults to ``False``.
        chunk_size (int): Sequence chunk of the teacher forward; the per-chunk logits tensor is
            ``[B, chunk_size, V]`` and the full ``[B, T, V]`` logits tensor is never materialized
            (§12 R5). Defaults to 512.
    """

    model_config = ConfigDict(extra="forbid")

    teacher: TrainTeacherConfig
    teacher_offload: bool = False
    chunk_size: int = Field(default=512, gt=0)


class QADTeacherAdapter:
    """Produce QAD teacher targets for packed SFT batches from a frozen BF16
    teacher.

    The adapter owns no model: the frozen teacher is passed to :meth:`compute_targets` by the
    training worker and executed synchronously in eval mode under ``torch.no_grad``. See the module
    docstring for the reuse/adaptation contract against the OPD ``TrainTeacherManager``.

    Args:
        config (QADTeacherAdapterConfig): The adapter configuration.
        student_vocab_size (int | TransformerConfig | BaseComposeConfig): The student vocabulary
            size, or the student model config to read it from. A mismatch with the teacher config
            raises ``ValueError`` at construction (§12 R7).
    """

    def __init__(
        self,
        config: QADTeacherAdapterConfig,
        student_vocab_size: int | TransformerConfig | BaseComposeConfig,
    ) -> None:
        self.config = config
        self.teacher_vocab_size = model_vocab_size(config.teacher.model_cfg)
        student_size = (
            student_vocab_size if isinstance(student_vocab_size, int) else model_vocab_size(student_vocab_size)
        )
        if student_size != self.teacher_vocab_size:
            raise ValueError(
                f"QAD teacher {config.teacher.name!r} vocab_size={self.teacher_vocab_size} does not match "
                f"student vocab_size={student_size}"
            )
        self.last_timings: TeacherTiming = TeacherTiming()

    def compute_targets(
        self,
        model_teacher: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        loss_cfg: QADLossConfig,
    ) -> dict[str, torch.Tensor]:
        """Compute the teacher distillation targets for one packed SFT batch.

        The teacher runs in eval mode under ``torch.no_grad``; with ``teacher_offload`` enabled it
        is onloaded before and offloaded after the forward (OPD cadence) and the lifecycle seconds
        are recorded in :attr:`last_timings`. The returned dict feeds
        :meth:`QADLossConfig.build` directly once the batch ``loss_mask`` is added.

        Args:
            model_teacher (torch.nn.Module): The frozen teacher; called as
                ``model(input_ids=..., position_ids=..., attention_mask=...)`` with the subset of
                fields its forward accepts, returning logits (tensor or object with ``.logits``).
            batch (dict[str, torch.Tensor]): Packed SFT batch; ``input_ids`` ``[B, T]`` is
                required, ``position_ids``/``attention_mask`` are forwarded when present. Rollout
                and response fields are neither required nor consumed.
            loss_cfg (QADLossConfig): The QAD loss configuration; ``mode`` selects the target
                contract, ``top_k`` and ``temperature`` the top-K selection and tempering.

        Returns:
            dict[str, torch.Tensor]: ``teacher_logprobs`` ``[B, T, K]`` plus ``target_token_ids``
            ``[B, T, K]`` for ``qad_topk_forward``, or ``teacher_logprobs`` ``[B, T, V]`` for
            ``qad_full_kl``; float32 (int64 for the ids), computed at the training temperature.
        """
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("QAD teacher targets require batch['input_ids'] [B, T]")
        if input_ids.dim() != 2 or input_ids.shape[1] == 0:
            raise ValueError(f"QAD teacher expects 2D non-empty input_ids [B, T], got {tuple(input_ids.shape)}")

        timings = TeacherTiming()
        self.last_timings = timings
        was_training = model_teacher.training
        model_teacher.eval()
        try:
            if self.config.teacher_offload:
                onload_begin = time.perf_counter()
                self._onload(model_teacher)
                timings.onload += time.perf_counter() - onload_begin
            try:
                compute_begin = time.perf_counter()
                targets = self._compute_targets(model_teacher, batch, loss_cfg)
                timings.compute += time.perf_counter() - compute_begin
            finally:
                if self.config.teacher_offload:
                    offload_begin = time.perf_counter()
                    self._offload(model_teacher)
                    timings.offload += time.perf_counter() - offload_begin
            return targets
        finally:
            if was_training:
                model_teacher.train()

    def _compute_targets(
        self,
        model_teacher: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        loss_cfg: QADLossConfig,
    ) -> dict[str, torch.Tensor]:
        input_ids = cast(torch.Tensor, batch["input_ids"])
        batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        is_topk = loss_cfg.mode == "qad_topk_forward"
        top_k = loss_cfg.top_k
        forward_fields = _select_forward_fields(model_teacher, batch)
        target_token_ids = (
            torch.empty((batch_size, seq_len, top_k), dtype=torch.int64, device=input_ids.device) if is_topk else None
        )
        teacher_logprobs: torch.Tensor | None = None
        with torch.no_grad():
            for start in range(0, seq_len, self.config.chunk_size):
                end = min(start + self.config.chunk_size, seq_len)
                chunk_kwargs = {name: tensor[:, start:end] for name, tensor in forward_fields.items()}
                logits = _forward_logits(model_teacher, chunk_kwargs)
                if logits.dim() != 3 or tuple(logits.shape[:2]) != (batch_size, end - start):
                    raise ValueError(
                        "QAD teacher forward must return logits [B, T_chunk, V] with "
                        f"B={batch_size} and T_chunk={end - start}, got {tuple(logits.shape)}"
                    )
                vocab_size = int(logits.shape[-1])
                # Tempering contract: divide by the training temperature before top-K and
                # log-softmax, in float32 (§7.2).
                tempered = logits.float() / loss_cfg.temperature
                if is_topk:
                    if vocab_size < top_k:
                        raise ValueError(f"top_k={top_k} exceeds the teacher vocabulary size {vocab_size}")
                    chunk_topk = torch.topk(tempered, k=top_k, dim=-1)
                    chunk_logprobs = torch.log_softmax(tempered, dim=-1).gather(dim=-1, index=chunk_topk.indices)
                    cast(torch.Tensor, target_token_ids)[:, start:end] = chunk_topk.indices
                else:
                    chunk_logprobs = torch.log_softmax(tempered, dim=-1)
                support_size = top_k if is_topk else vocab_size
                if teacher_logprobs is None:
                    teacher_logprobs = torch.empty(
                        (batch_size, seq_len, support_size), dtype=torch.float32, device=chunk_logprobs.device
                    )
                teacher_logprobs[:, start:end] = chunk_logprobs
        targets: dict[str, torch.Tensor] = {"teacher_logprobs": cast(torch.Tensor, teacher_logprobs)}
        if target_token_ids is not None:
            targets["target_token_ids"] = target_token_ids
        return targets

    def _onload(self, model_teacher: torch.nn.Module) -> None:
        _move_model(model_teacher, DEVICE)

    def _offload(self, model_teacher: torch.nn.Module) -> None:
        _move_model(model_teacher, "cpu")
        if hasattr(DEVICE_MODULE, "empty_cache"):
            DEVICE_MODULE.empty_cache()


def _forward_logits(model_teacher: torch.nn.Module, chunk_kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model_teacher(**chunk_kwargs)
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            f"the QAD teacher forward must return logits or an object with a tensor .logits, got {type(output)!r}"
        )
    return logits


def _select_forward_fields(
    model_teacher: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    try:
        parameters = inspect.signature(model_teacher.forward).parameters.values()
    except (TypeError, ValueError):
        return {name: batch[name] for name in FORWARD_KEYS if name in batch}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return {name: batch[name] for name in FORWARD_KEYS if name in batch}
    accepted = {parameter.name for parameter in parameters}
    return {name: batch[name] for name in FORWARD_KEYS if name in accepted and name in batch}


def _move_model(model_teacher: torch.nn.Module, device: torch.device | str) -> None:
    to_device = getattr(model_teacher, "to_device", None)
    if callable(to_device):
        cast(Callable[[str], object], to_device)(str(device))
    else:
        model_teacher.to(device=device)
