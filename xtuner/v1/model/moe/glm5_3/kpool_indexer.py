# Copyright (c) OpenMMLab. All rights reserved.
"""K-pool compressed DSA indexer for GLM-5.3-Flash.

Port of ``Glm5NextKPoolIndexer`` (NVIDIA Automodel) /
``Glm5NextTextIndexer`` (transformers) onto XTuner's SequenceContext contract.

Differences from the per-token DSA indexer (``glm52.dsa_mla.DSAIndexer``):

* keys are grouped into pools of ``index_kpool`` consecutive tokens and the
  indexer scores *pools* — every token of a selected pool becomes a sparse
  attention candidate, expanding the effective budget ``index_topk`` by up to
  ``index_kpool``× at ``1 / index_kpool`` of the scoring cost;
* pool compression is a softmax-gated mean: ``softmax(gates + ape)`` weights
  the pooled keys, where ``gates`` comes from a learned per-channel projection
* with ``index_kpool_always_select_tail`` the (at most ``index_kpool - 1``)
  tokens of the trailing incomplete pool of each query's visible prefix are
  always appended, so the newest tokens are never missed by pool selection.

Selection runs under ``no_grad``: indices are non-differentiable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import TypedDict

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.linear import build_linear


class KPoolDSAIndexerOutput(TypedDict):
    dsa_topk_ids: torch.Tensor


class KPoolDSAIndexer(nn.Module):
    """Pooled DSA indexer over packed varlen documents.

    Args:
        hidden_size (int): Model hidden size.
        q_lora_rank (int): Query LoRA rank feeding ``wq_b``.
        index_head_dim (int): Indexer head dimension.
        index_n_heads (int): Indexer head count.
        index_topk (int): Nominal sparse-attention budget per query.
        index_kpool (int): Tokens per selection pool.
        index_kpool_always_select_tail (bool): Append the always-visible tail
            of each query's visible prefix after the selected pools.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        index_head_dim: int,
        index_n_heads: int,
        index_topk: int,
        index_kpool: int,
        index_kpool_always_select_tail: bool,
    ) -> None:
        super().__init__()
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.softmax_scale = index_head_dim**-0.5

        self.wq_b = build_linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
        self.wk = build_linear(hidden_size, index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(index_head_dim, eps=1e-6)
        self.weights_proj = build_linear(hidden_size, index_n_heads, bias=False)
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(index_kpool, index_head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(index_head_dim, hidden_size))

    def custom_init_weights(self) -> set[str]:
        """Zero APE, fill the compress gate with ones (HF semantics)."""
        torch.nn.init.zeros_(self.index_kpool_compress_ape)
        torch.nn.init.ones_(self.index_kpool_compress_gate)
        return {"index_kpool_compress_ape", "index_kpool_compress_gate"}

    @torch.no_grad()
    def _prepare_pools(self, keys: torch.Tensor, gates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress one document's keys into pool candidates.

        Args:
            keys (torch.Tensor): Indexer keys ``[T, head_dim]`` (one document).
            gates (torch.Tensor): Gate logits ``[T, head_dim]``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(pool_keys, pool_indices)``
            with shapes ``[P, head_dim]`` and ``[P, kpool]``; ``P = T // kpool``.
        """
        kp = self.index_kpool
        length = keys.shape[0]
        complete_pools = length // kp
        if complete_pools == 0:
            empty_keys = keys.new_empty((0, self.index_head_dim))
            empty_idx = torch.empty((0, kp), dtype=torch.long, device=keys.device)
            return empty_keys, empty_idx

        width = complete_pools * kp
        grouped_keys = keys[:width].view(complete_pools, kp, self.index_head_dim)
        grouped_gates = gates[:width].view(complete_pools, kp, self.index_head_dim)
        logits = grouped_gates.float() + self.index_kpool_compress_ape.float().unsqueeze(0)
        pool_keys = (logits.softmax(dim=1).to(keys.dtype) * grouped_keys).sum(dim=1)
        pool_indices = torch.arange(width, device=keys.device).view(complete_pools, kp)
        return pool_keys, pool_indices

    @torch.no_grad()
    def _select(
        self,
        query_hidden: torch.Tensor,
        query_resid: torch.Tensor,
        query_positions: torch.Tensor,
        pool_keys: torch.Tensor,
        pool_indices: torch.Tensor,
        key_length: int,
    ) -> torch.Tensor:
        """Pick raw key indices for one query chunk of one document.

        Args:
            query_hidden (torch.Tensor): ``[1, Q, hidden]``.
            query_resid (torch.Tensor): ``[1, Q, q_lora]``.
            query_positions (torch.Tensor): Document-local positions ``[Q]``.
            pool_keys (torch.Tensor): ``[P, head_dim]``.
            pool_indices (torch.Tensor): ``[P, kpool]``.
            key_length (int): Document key length.

        Returns:
            torch.Tensor: int32 indices ``[1, Q, topk (+ kpool - 1)]``.
        """
        del key_length
        complete_pools = pool_keys.shape[0]

        queries = self.wq_b(query_resid).view(1, -1, self.index_n_heads, self.index_head_dim)
        scores = torch.einsum("bqhd,pd->bqhp", queries.float(), pool_keys.float())
        scores = F.relu(scores * self.softmax_scale)
        weights = self.weights_proj(query_hidden).float() * (self.index_n_heads**-0.5)
        scores = torch.einsum("bqh,bqhp->bqp", weights, scores)

        if complete_pools:
            pool_idx = torch.arange(complete_pools, device=query_hidden.device)
            pool_end = pool_indices[:, -1]
            visible = pool_end.view(1, 1, -1) <= query_positions.view(1, -1, 1)
            # The pool containing the query itself is always visible (self is
            # selectable); this keeps softmax non-degenerate for early queries
            # whose complete-pool count is below ``select_k``.
            own_pool = torch.div(query_positions, self.index_kpool, rounding_mode="floor")
            visible = visible | (pool_idx.view(1, 1, -1) == own_pool.view(1, -1, 1))
            scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
            select_k = min(self.index_topk // self.index_kpool, complete_pools)
            selected = scores.topk(select_k, dim=-1).indices
            selected_valid = visible.expand_as(scores).gather(-1, selected)
            raw = pool_indices[selected].flatten(-2)
            raw = raw.masked_fill(~selected_valid.unsqueeze(-1).expand_as(pool_indices[selected]).flatten(-2), -1)
        else:
            raw = torch.empty((1, query_hidden.shape[1], 0), dtype=torch.long, device=query_hidden.device)

        output_width = self.index_topk
        if self.index_kpool_always_select_tail and self.index_kpool > 1:
            tail_count = (query_positions + 1).remainder(self.index_kpool)
            tail_start = query_positions + 1 - tail_count
            offsets = torch.arange(self.index_kpool - 1, device=query_hidden.device)
            tail = tail_start[:, None] + offsets
            tail = tail.masked_fill(offsets[None] >= tail_count[:, None], -1).unsqueeze(0)
            raw = torch.cat((raw, tail), dim=-1)
            output_width += self.index_kpool - 1
        return F.pad(raw, (0, max(output_width - raw.shape[-1], 0)), value=-1)[..., :output_width].to(torch.int32)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        seq_ctx: SequenceContext,
    ) -> KPoolDSAIndexerOutput:
        """Compute pooled top-k indices for every query.

        Processes each varlen document independently (boundaries from
        ``seq_ctx.cu_seq_lens_q_cpu``), then writes the per-document results
        back into the packed ``[B, S, ...]`` layout.

        Args:
            hidden_states (torch.Tensor): ``[B, S, hidden]`` layer input.
            q_resid (torch.Tensor): ``[B, S, q_lora]`` post-norm query residual.
            position_embeddings (tuple | None): Unused — the GLM indexer runs
                on rope-free nope channels; kept for call-site uniformity.
            seq_ctx (SequenceContext): Sequence context providing varlen
                boundaries.

        Returns:
            KPoolDSAIndexerOutput: ``dsa_topk_ids`` int32 ``[B, S, topk(+tail)]``.
        """
        del position_embeddings
        bsz, seq_len, _ = hidden_states.shape

        keys = self.k_norm(self.wk(hidden_states))
        gates = F.linear(hidden_states, self.index_kpool_compress_gate)
        # upstream/main's SequenceContext keeps cu_seq_lens on the GPU only;
        # indexer selection runs document-by-document on CPU scalars, so pay
        # one explicit sync here (inside no_grad, off the autograd path).
        cu = seq_ctx.cu_seq_lens_q
        if cu is not None:
            cu = cu.detach().to("cpu", torch.int64) if cu.is_cuda else cu
        if cu is not None and cu.numel() >= 2:
            bounds = cu.flatten().tolist()
        else:
            bounds = [0, seq_len]

        out_width = self.index_topk + (self.index_kpool - 1 if self.index_kpool_always_select_tail else 0)
        out = torch.full((bsz, seq_len, out_width), -1, dtype=torch.int32, device=hidden_states.device)
        for d in range(len(bounds) - 1):
            start, end = bounds[d], bounds[d + 1]
            if end <= start:
                continue
            doc_len = end - start
            positions = torch.arange(doc_len, device=hidden_states.device)
            doc_keys = keys[0, start:end]
            doc_gates = gates[0, start:end]
            pool_keys, pool_indices = self._prepare_pools(doc_keys, doc_gates)
            ids = self._select(
                query_hidden=hidden_states[:, start:end],
                query_resid=q_resid[:, start:end],
                query_positions=positions,
                pool_keys=pool_keys,
                pool_indices=pool_indices,
                key_length=doc_len,
            )
            # _select emits document-local token indices; shift to global.
            ids = torch.where(ids >= 0, ids + start, ids)
            out[:, start:end] = ids
        return {"dsa_topk_ids": out}
