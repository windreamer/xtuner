# Copyright (c) OpenMMLab. All rights reserved.
"""Quantization-aware distillation (QAD) loss.

Implements the §7.2 loss registry of the QAD design: a single forward-KL kernel shared by two
target contracts. ``qad_topk_forward`` consumes the OPD Top-K teacher contract
(``target_token_ids [B, T, K]`` + ``teacher_logprobs [B, T, K]``) and reproduces
``compute_topk_distillation_kl`` in forward mode exactly; ``qad_full_kl`` consumes full-vocabulary
teacher logprobs (``teacher_logprobs [B, T, V]``, assembled per micro-batch by the trainer) and
feeds them through the same kernel, where the tail bin vanishes at ``K = V``.

Temperature contract: the teacher runtime must compute its targets at the configured temperature
(teacher divides its logits by ``T`` before top-K selection / log-softmax); this context divides
the student logits by ``T`` and scales the per-token loss by ``T^2`` (standard KD convention).
"""

from typing import Any, Literal, Sequence, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import Field
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce
from typing_extensions import Self

from xtuner.v1.loss.base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs
from xtuner.v1.loss.utils import sp_split
from xtuner.v1.rl.loss.distillation_loss import compute_topk_distillation_kl
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.device import get_device


DEVICE = get_device()
logger = get_logger()

QADLossMode = Literal["qad_topk_forward", "qad_full_kl"]


def compute_qad_forward_kl(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    log_prob_min_clamp: float,
    loss_max_clamp: float,
) -> torch.Tensor:
    """Compute the shared QAD forward-KL kernel over a selected token support.

    This is the single KL implementation behind both QAD loss modes. It re-uses the OPD
    top-K forward KL: the teacher-selected support carries per-token probabilities, the
    remaining probability mass is folded into one tail bin per token, and the loss is
    ``sum_topk q(log q - log p) + q_tail(log q_tail - log p_tail)``. For the full-vocabulary
    mode the support is the whole vocabulary, both tail bins vanish, and the kernel reduces
    to the exact full-vocabulary forward KL. ``log_prob_min_clamp`` is applied to both sides
    before the KL and ``loss_max_clamp`` to the per-token loss afterwards (the OPD clamp
    placement); the OPD ``"forward"`` branch itself leaves the log-probabilities untouched,
    so the clamp is applied here.

    Args:
        student_logprobs (torch.Tensor): Student log-probabilities on the selected support,
            shape ``[B, T, S]`` with ``S = K`` (top-K mode) or ``S = V`` (full-vocabulary mode).
        teacher_logprobs (torch.Tensor): Teacher log-probabilities on the same support,
            shape ``[B, T, S]``, computed at the training temperature.
        log_prob_min_clamp (float): Lower clamp applied to both student and teacher
            log-probabilities before the KL term.
        loss_max_clamp (float): Upper clamp applied to the per-token loss afterwards.

    Returns:
        torch.Tensor: Per-token forward KL of shape ``[B, T]``.
    """
    student_logprobs = student_logprobs.clamp_min(log_prob_min_clamp)
    teacher_logprobs = teacher_logprobs.clamp_min(log_prob_min_clamp)
    loss, _, _ = compute_topk_distillation_kl(
        student_logprobs,
        teacher_logprobs,
        "forward",
        log_prob_min_clamp=None,
        loss_max_clamp=loss_max_clamp,
    )
    return loss


class QADLossConfig(BaseLossConfig):
    """Configuration of the quantization-aware distillation loss.

    Args:
        ignore_idx (int): The index to ignore in the loss calculation. Defaults to -100.
        mode (str): The QAD loss mode. ``"qad_topk_forward"`` consumes the OPD Top-K teacher
            contract (``target_token_ids [B, T, K]`` + ``teacher_logprobs [B, T, K]``);
            ``"qad_full_kl"`` consumes full-vocabulary teacher logprobs
            (``teacher_logprobs [B, T, V]``). Defaults to ``"qad_topk_forward"``.
        chunk_size (int | None): Chunk size for chunked loss calculation. Unused by this loss.
            Defaults to 1024.
        top_k (int): Number of teacher-selected target tokens of the ``qad_topk_forward``
            contract. Consumed by the teacher runtime that produces the targets; the loss
            itself follows the ``K`` carried by ``target_token_ids``. Defaults to 64.
        temperature (float): KD temperature. ``1.0`` matches the teacher distribution exactly;
            values other than ``1.0`` scale the student logits by ``1 / T`` and the per-token
            loss by ``T ** 2``. Defaults to 1.0.
        log_prob_min_clamp (float): Lower clamp applied to student and teacher log-probabilities
            before the KL. Defaults to -10.0.
        loss_max_clamp (float): Upper clamp applied to the per-token loss. Must be positive.
            Defaults to 10.0.
    """

    mode: QADLossMode = "qad_topk_forward"  # type: ignore[assignment]
    top_k: int = Field(default=64, gt=0)
    temperature: float = Field(default=1.0, gt=0.0)
    log_prob_min_clamp: float = -10.0
    loss_max_clamp: float = Field(default=10.0, gt=0.0)

    @property
    def loss_ctx_cls(self) -> type["QADLossContext"]:
        return QADLossContext

    @property
    def _loss_kwargs_cls(self) -> type["QADLossKwargs"]:
        return QADLossKwargs

    def build(
        self,
        data: dict,
        sp_mesh: DeviceMesh | None = None,
        device: torch.device | str | None = None,
    ) -> "QADLossContext | None":
        """Build a local QAD loss context from complete, unsplit data.

        The data dict carries the packed batch ``loss_mask`` [B, T] and the teacher targets
        (``teacher_logprobs`` and, for top-K mode, ``target_token_ids``). Teacher targets must
        be computed at the configured temperature. If a sequence parallel mesh is given, the
        loss kwargs are sharded along the sequence dimension before being moved to the device.

        Args:
            data (dict): Data dict containing ``loss_mask``, ``teacher_logprobs`` and, for
                top-K mode, ``target_token_ids``.
            sp_mesh (DeviceMesh | None): Sequence parallel mesh.
            device (torch.device | str | None): Target device of the returned context;
                ``None`` resolves to the implementation default device.

        Returns:
            QADLossContext | None: The local loss context, or ``None`` if ``data`` carries no
            ``teacher_logprobs`` (i.e. it is not a QAD batch).
        """
        if "teacher_logprobs" not in data or "loss_mask" not in data:
            return None
        if self.mode == "qad_topk_forward" and data.get("target_token_ids") is None:
            raise ValueError('mode="qad_topk_forward" requires target_token_ids [B, T, K] in data')
        if device is None:
            device = DEVICE
        loss_kwargs = QADLossKwargs(
            loss_mask=data["loss_mask"],
            teacher_logprobs=data["teacher_logprobs"],
            target_token_ids=data.get("target_token_ids"),
        )
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)
        return self.loss_ctx_cls(self, loss_kwargs.to(device))


class QADLossKwargs(BaseLossKwargs):
    """Keyword arguments of the QAD loss.

    Args:
        loss_mask (torch.Tensor): Per-token loss mask of shape ``[B, T]``; positions with a
            non-zero value contribute to the loss reduction.
        teacher_logprobs (torch.Tensor): Teacher log-probabilities at the training temperature,
            ``[B, T, K]`` on the teacher-selected support (top-K mode) or ``[B, T, V]``
            (full-vocabulary mode).
        target_token_ids (torch.Tensor | None): Teacher-selected token ids of shape
            ``[B, T, K]`` (int64); required by top-K mode, ``None`` otherwise.
        loss_weight (torch.Tensor | None): Per-token weight ``loss_mask / global_mask_sum``
            calibrated by ``QADLossContext.build_batches``; ``None`` before calibration.
    """

    loss_mask: torch.Tensor
    teacher_logprobs: torch.Tensor
    target_token_ids: torch.Tensor | None = None
    loss_weight: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> Self:
        """Shard every tensor field along the sequence dimension for sequence
        parallelism.

        Args:
            sp_mesh (DeviceMesh): Sequence parallel mesh; padded shards are zero-filled so
                that the calibrated loss weight masks them out.

        Returns:
            QADLossKwargs: The sharded loss kwargs.
        """
        self.loss_mask = sp_split(self.loss_mask, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        self.teacher_logprobs = sp_split(self.teacher_logprobs, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        if self.target_token_ids is not None:
            self.target_token_ids = sp_split(self.target_token_ids, sp_mesh=sp_mesh, split_dim=1, padding_value=0)
        if self.loss_weight is not None:
            self.loss_weight = sp_split(self.loss_weight, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        return self

    def to(self, device: torch.device | str) -> Self:
        """Move every tensor field to the target device.

        Args:
            device (torch.device | str): Target device.

        Returns:
            QADLossKwargs: The loss kwargs on the target device.
        """
        self.loss_mask = self.loss_mask.to(device)
        self.teacher_logprobs = self.teacher_logprobs.to(device)
        if self.target_token_ids is not None:
            self.target_token_ids = self.target_token_ids.to(device)
        if self.loss_weight is not None:
            self.loss_weight = self.loss_weight.to(device)
        return self


class QADLossContext(BaseLossContext):
    """Loss context of the quantization-aware distillation loss.

    Computes a direct-backprop forward KL between the student output distribution and the
    teacher targets, reduced to a per-token mean over ``loss_mask`` through the shared
    dp/sp/grad-accumulation calibration of ``build_batches``.

    Args:
        loss_cfg (QADLossConfig): The QAD loss configuration.
        loss_kwargs (QADLossKwargs): The teacher targets and loss mask of this micro-batch.
    """

    loss_cfg: QADLossConfig
    loss_kwargs: QADLossKwargs

    def __init__(self, loss_cfg: QADLossConfig, loss_kwargs: QADLossKwargs):
        super().__init__(loss_cfg, loss_kwargs)

    @staticmethod
    def build_batches(  # type: ignore[override]
        loss_ctx_list: list["QADLossContext"],
        cu_seq_lens_list: Sequence[torch.IntTensor] | None = None,
        sp_mesh: DeviceMesh | None = None,
    ) -> list["QADLossContext"]:
        """Calibrate one optimizer step's contexts for per-token mean reduction
        over loss_mask.

        Following the dp/sp/grad-accumulation calibration protocol, the global loss mask sum
        is accumulated over every micro-batch of the step and across all ranks; each context
        then receives the per-token weight ``loss_mask / global_loss_mask_sum`` so that the
        sum of the weighted micro-batch losses equals the global masked mean.

        Args:
            loss_ctx_list (list[QADLossContext]): Contexts of one optimizer step.
            cu_seq_lens_list (Sequence[torch.IntTensor] | None): Unused; kept for the shared
                build protocol.
            sp_mesh (DeviceMesh | None): Unused; kept for the shared build protocol.

        Returns:
            list[QADLossContext]: The same contexts with calibrated per-token weights.
        """
        del cu_seq_lens_list, sp_mesh
        assert len(loss_ctx_list) > 0, "loss_ctx_list can not be empty"

        loss_mask_list = [loss_ctx.loss_kwargs.loss_mask for loss_ctx in loss_ctx_list]
        rank_denominator = sum(mask.sum() for mask in loss_mask_list)
        global_denominator = cast(torch.Tensor, rank_denominator)
        if dist.is_initialized():
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
        if global_denominator == 0:
            logger.warning("Global loss mask sum is 0; using one as the loss denominator")
            global_denominator = global_denominator + 1.0

        for loss_ctx in loss_ctx_list:
            loss_ctx._batch_size = len(loss_ctx_list)
            loss_mask = loss_ctx.loss_kwargs.loss_mask
            if not loss_mask.is_floating_point():
                loss_mask = loss_mask.to(torch.float32)
            loss_ctx.loss_kwargs.loss_weight = loss_mask / global_denominator
        return loss_ctx_list

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        """Compute the calibrated QAD loss from the student final hidden
        states.

        Args:
            hidden_states (torch.Tensor): Student final hidden states of shape ``[B, T, D]``.
            head_weight (torch.Tensor): Language-model head weight of shape ``[V, D]``.
            head_bias (torch.Tensor | None): Language-model head bias; ``None`` when the head
                has no bias.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]: The calibrated
            scalar loss and a tuple of the student logits and the reduced metric dict.
        """
        logits = F.linear(hidden_states, head_weight, head_bias)
        logits = _promote_logits_dtype(logits)
        temperature = self.loss_cfg.temperature
        if temperature != 1.0:
            logits = logits / temperature
        teacher_logprobs = self.loss_kwargs.teacher_logprobs.to(logits.dtype)

        if self.loss_cfg.mode == "qad_topk_forward":
            per_token_loss, metrics = self._compute_topk_forward_loss(logits, teacher_logprobs)
        else:
            student_logprobs = torch.log_softmax(logits, dim=-1)
            per_token_loss = compute_qad_forward_kl(
                student_logprobs,
                teacher_logprobs,
                self.loss_cfg.log_prob_min_clamp,
                self.loss_cfg.loss_max_clamp,
            )
            metrics = {}
        if temperature != 1.0:
            per_token_loss = per_token_loss * temperature**2
        metrics["reduced_qad_kl_sum"] = (per_token_loss.detach() * self.loss_kwargs.loss_mask).sum()
        metrics["reduced_qad_valid_count"] = self.loss_kwargs.loss_mask.sum()

        loss = self._reduce(per_token_loss)
        metrics["local_base_loss"] = loss.detach().clone()
        if dist.is_initialized():
            loss = all_reduce(loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        return loss, (logits, metrics)

    def _reduce(self, per_token_loss: torch.Tensor) -> torch.Tensor:
        loss_weight = self.loss_kwargs.loss_weight
        assert loss_weight is not None, "loss_weight must be calibrated by build_batches before forward"
        return (per_token_loss * loss_weight).sum()

    def _compute_topk_forward_loss(
        self,
        logits: torch.Tensor,
        teacher_logprobs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        target_token_ids = self.loss_kwargs.target_token_ids
        assert target_token_ids is not None, "qad_topk_forward requires target_token_ids in loss kwargs"

        student_topk_logprobs = torch.gather(logits, dim=-1, index=target_token_ids)
        student_topk_logprobs = student_topk_logprobs - torch.logsumexp(logits, dim=-1, keepdim=True)

        per_token_loss = compute_qad_forward_kl(
            student_topk_logprobs,
            teacher_logprobs,
            self.loss_cfg.log_prob_min_clamp,
            self.loss_cfg.loss_max_clamp,
        )

        student_topk_ids = torch.topk(logits.detach(), k=target_token_ids.size(-1), dim=-1).indices
        overlap = (student_topk_ids.unsqueeze(-1) == target_token_ids.detach().unsqueeze(-2)).any(dim=-1)
        overlap_fraction = overlap.float().mean(dim=-1)
        metrics: dict[str, Any] = {
            "reduced_qad_topk_overlap_fraction_sum": (overlap_fraction * self.loss_kwargs.loss_mask).sum().detach()
        }
        return per_token_loss, metrics


def _promote_logits_dtype(logits: torch.Tensor) -> torch.Tensor:
    if logits.dtype in (torch.float16, torch.bfloat16):
        return logits.float()
    return logits
