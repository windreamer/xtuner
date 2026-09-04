# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash text model: mHC decoder layer over a hybrid KDA / DSA stack.

The text tower follows the ``MoE`` skeleton (embed → decoder stack → norm →
lm_head). Every decoder layer keeps ``hc_mult`` persistent residual streams;
two mHC sites (attention, feed-forward) collapse the streams into the
sub-layer input and re-expand the sub-layer output back onto the streams via
the Sinkhorn-projected ``comb`` mixer. Attention alternates KDA linear
attention (``layer_types[i] == "linear_attention"``) with KPool-DSA sparse
MLA (``"deepseek_sparse_attention"``); the feed-forward is dense for the
first ``first_k_dense_replace`` layers and noaux-sigmoid MoE afterwards.
"""

import torch
import torch.nn as nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.glm5_3.kpool_indexer import KPoolDSAIndexer
from xtuner.v1.model.moe.glm5_3.mhc import GlmMHC, GlmMHCHead


__all__ = ["Glm53DecoderLayer"]


class Glm53DecoderLayer(nn.Module):
    """One mHC decoder block: KDA/DSA attention site + dense/MoE FFN site.

    The heavy sub-layers (MLA sparse attention, MoE FFN) are injected by the
    model's ``build_layers`` so this layer stays a thin composition of the
    two mHC wrappers and two sub-layers — mirroring
    ``Glm5NextDecoderLayer``.
    """

    def __init__(
        self,
        *,
        layer_idx: int,
        layer_type: str,
        is_moe: bool,
        hidden_size: int,
        hc_mult: int,
        hc_eps: float,
        hc_sinkhorn_iters: int,
        rms_norm_eps: float,
        self_attn: nn.Module,
        mlp: nn.Module,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.is_moe_layer = is_moe
        self.self_attn = self_attn
        self.mlp = mlp

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attn_hc = GlmMHC(
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            hc_eps=hc_eps,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            rms_norm_eps=rms_norm_eps,
        )
        self.ffn_hc = GlmMHC(
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            hc_eps=hc_eps,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            rms_norm_eps=rms_norm_eps,
        )

    @staticmethod
    def _mhc_post(
        update: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Re-expand the sub-layer output onto the residual streams.

        Args:
            update (torch.Tensor): Sub-layer output ``[B, S, D]``.
            post (torch.Tensor): Stream placement weights ``[B, S, H]``.
            comb (torch.Tensor): Stream mixer ``[B, S, H, H]``.
            residual (torch.Tensor): Incoming streams ``[B, S, H, D]``.

        Returns:
            torch.Tensor: Updated streams ``[B, S, H, D]``.
        """
        dtype = residual.dtype
        return post.to(dtype).unsqueeze(-1) * update.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

    def forward(
        self,
        hidden_streams: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        """Transform the residual streams through one layer.

        Args:
            hidden_streams (torch.Tensor): ``[B, S, hc_mult, D]``.
            position_embeddings (tuple | None): Dense rope ``(cos, sin)``.
            seq_ctx (SequenceContext): Sequence context (varlen boundaries,
                k-pool caches).

        Returns:
            torch.Tensor: Updated streams ``[B, S, hc_mult, D]``.
        """
        residual = hidden_streams
        collapsed, post, comb = self.attn_hc(hidden_streams)
        update = self.self_attn(self.input_layernorm(collapsed), position_embeddings, seq_ctx)
        hidden_streams = self._mhc_post(update, post, comb, residual)

        residual = hidden_streams
        collapsed, post, comb = self.ffn_hc(hidden_streams)
        update = self.post_attention_layernorm(collapsed)
        update = self.mlp(update)
        return self._mhc_post(update, post, comb, residual)


class Glm53HyperHead(GlmMHCHead):
    """Re-exported for symmetry with the model file (unweighted mean)."""


class Glm53FlashSparseAttention(nn.Module):
    """KPool-indexed sparse MLA with ``qk_rope_head_dim = 0`` (nope-only).

    Low-rank attention (q LoRA + kv LoRA) with an absorbed forward: the query
    nope half is absorbed by ``kv_b_proj``'s key half so attention runs in the
    512-wide compressed-KV basis over the KPool-selected candidates.

    Args:
        hidden_size (int): Model hidden size.
        num_attention_heads (int): Query head count.
        q_lora_rank (int): Query LoRA rank.
        kv_lora_rank (int): Compressed KV rank.
        qk_nope_head_dim (int): Query/key nope head dimension.
        v_head_dim (int): Value head dimension.
        index_head_dim (int): Indexer head dimension.
        index_n_heads (int): Indexer head count.
        index_topk (int): Sparse-attention budget per query.
        index_kpool (int): Tokens per indexer selection pool.
        index_kpool_always_select_tail (bool): Always append the visible tail.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        index_head_dim: int,
        index_n_heads: int,
        index_topk: int,
        index_kpool: int,
        index_kpool_always_select_tail: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.softmax_scale = qk_nope_head_dim**-0.5

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(q_lora_rank, num_attention_heads * qk_nope_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim), bias=False)
        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)

        self.indexer = KPoolDSAIndexer(
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_topk=index_topk,
            index_kpool=index_kpool,
            index_kpool_always_select_tail=index_kpool_always_select_tail,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,  # unused: nope-only
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        """Absorbed sparse-MLA forward over KPool-selected candidates.

        Args:
            hidden_states (torch.Tensor): ``[B, S, D]`` collapsed stream.
            position_embeddings (tuple | None): Unused (GLM-5.3 DSA has no
                decoupled rope: ``qk_rope_head_dim == 0``).
            seq_ctx (SequenceContext): Sequence context (varlen boundaries).

        Returns:
            torch.Tensor: ``[B, S, D]`` attention output.
        """
        del position_embeddings
        bsz, q_len, _ = hidden_states.size()

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_nope = self.q_b_proj(q_resid).view(bsz, q_len, self.num_attention_heads, self.qk_nope_head_dim)

        kv_c = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
        # NOTE: ``kv_b_proj`` is applied inside the absorbed path below (W_uk /
        # W_uv views of its weight), so no eager projection is needed here.

        topk = self.indexer(hidden_states, q_resid, None, seq_ctx)["dsa_topk_ids"]  # [B, S, K]

        # Absorb: q_nope @ W_uk where W_uk = kv_b key half → attention in the
        # compressed-KV basis over gathered candidates.
        w_uk, w_uv = self.kv_b_proj.weight.split(
            [self.qk_nope_head_dim * self.num_attention_heads, self.v_head_dim * self.num_attention_heads], dim=0
        )
        w_uk = w_uk.view(self.num_attention_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        w_uv = w_uv.view(self.num_attention_heads, self.kv_lora_rank, self.v_head_dim)

        q_abs = torch.einsum("bqhd,hdk->bhqk", q_nope, w_uk)  # [B, S, H, Rkv]

        # Gathered attention over each query's candidates (padded with -1).
        B_, S_, K = topk.shape
        flat_idx = topk.clamp(min=0).long()
        kv_c_exp = kv_c.unsqueeze(2)  # [B, S, 1, Rkv]
        gathered = torch.gather(
            kv_c_exp.expand(-1, -1, K, -1), 1, flat_idx.unsqueeze(-1).expand(-1, -1, -1, self.kv_lora_rank)
        )  # [B, S, K, Rkv]
        attn = torch.einsum("bhsr,bscr->bhsc", q_abs, gathered) * self.softmax_scale  # [B, H, S, K]
        valid = (topk >= 0).unsqueeze(1)
        attn = attn.masked_fill(~valid, torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        # Absorbed output: the attention is over the compressed-KV basis, so
        # the context is per-head ``[B, H, S, Rkv]``; project through W_uv
        # (per-head) to the value head dim afterwards.
        context = torch.einsum("bhsc,bscr->bhsr", attn, gathered)  # [B, H, S, Rkv]
        out = torch.einsum("bhsk,hkv->bhsv", context, w_uv)  # [B, H, S, V]
        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_attention_heads * self.v_head_dim)
        return self.o_proj(out)
