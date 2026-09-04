# Copyright (c) OpenMMLab. All rights reserved.
"""KDA (Kimi Delta Attention) linear attention for GLM-5.3-Flash.

Ports ``transformers.models.glm5_next.Glm5NextTextLinearAttention`` onto
XTuner's ``AttnOutputs`` contract so a hybrid KDA / DSA stack can share the
``MoE`` decoder skeleton.

The recurrence kernel is fla's ``chunk_kda`` (training/prefill) and
``fused_recurrent_kda`` (decode); both are wrapped behind
:func:`get_kda_core_fn` so parity tests can force the HF-exact torch fallback
without touching call sites — the same seam discipline as
``xtuner.v1.ops.attn_imp``.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8 import Float8Config
from xtuner.v1.module.attention.gated_deltanet import FusedRMSNormGated

from .attn_imp import get_kda_core_fn


class KdaLinearAttentionConfig(BaseModel):
    model_config = {"extra": "forbid"}

    num_heads: int
    head_dim: int
    conv_kernel_dim: int = 4
    gate_lower_bound: float | None = None
    rms_norm_eps: float = 1e-5
    layer_type: Literal["linear_attention"] = "linear_attention"

    def build(
        self,
        hidden_size: int,
        layer_idx: int = 0,
        float8_cfg: Float8Config | None = None,
        **kwargs,
    ) -> "KdaLinearAttention":
        return KdaLinearAttention(
            **self.model_dump(exclude=("layer_type",)),
            hidden_size=hidden_size,
            layer_idx=layer_idx,
        )


class KdaForgetGate(nn.Module):
    """Per-head decay gate: ``-decay_rate * softplus(g)`` or the safe
    ``lower_bound * sigmoid(decay_rate * g)`` variant."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        gate_lower_bound: float | None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.qkv_dim = head_dim * num_heads

        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.safe_gate_lower_bound = gate_lower_bound

    def custom_init_weights(self) -> set[str]:
        """HF-matching init for the fp32 decay parameters."""
        torch.nn.init.zeros_(self.dt_bias)
        torch.nn.init.ones_(self.A_log)
        return {"dt_bias", "A_log"}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)

        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        A_log = self.A_log.float().view(1, 1, self.num_heads, 1)
        decay_rate = torch.exp(A_log)

        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)

        # Softplus with the >20 overflow guard (softplus(x) == x for x > 20).
        g_softplus = torch.where(g > 20.0, g, torch.log1p(torch.exp(g)))
        return -decay_rate * g_softplus


class KdaLinearAttention(nn.Module):
    """GLM-5.3-Flash KDA layer.
    PLACEHOLDER_FIELDS
            hidden_size (int): Model hidden size.
            num_heads (int): Linear-attention head count.
            head_dim (int): Per-head key/value dimension.
            conv_kernel_dim (int): Short causal conv kernel for q/k/v.
            gate_lower_bound (float | None): Safe lower bound for the decay gate;
                ``None`` selects the softplus decay variant.
            rms_norm_eps (float): Epsilon for the output gated RMSNorm.
            layer_idx (int): Decoder layer position (naming only).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_dim: int = 4,
        gate_lower_bound: float | None = None,
        rms_norm_eps: float = 1e-5,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_dim = head_dim * num_heads
        self.conv_kernel_dim = conv_kernel_dim
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.qkv_dim, bias=False)

        # One depthwise causal conv over the concatenated q/k/v, exactly like
        # HF: a single Conv1d with groups=qkv_dim*3 and no bias.
        conv_dim = self.qkv_dim * 3
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=False,
            kernel_size=conv_kernel_dim,
            groups=conv_dim,
            padding=conv_kernel_dim - 1,
        )

        self.forget_gate = KdaForgetGate(hidden_size, num_heads, head_dim, gate_lower_bound)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, self.qkv_dim, bias=False)
        self.o_norm = FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="silu")
        self.o_proj = nn.Linear(self.qkv_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # unused: rope-free
        seq_ctx: SequenceContext | None = None,
    ) -> torch.Tensor:
        """KDA forward (training / full-prefill path).

        Args:
            hidden_states (torch.Tensor): Collapsed input stream ``[B, S, D]``.
            position_embeddings: Unused (KDA is rope-free); kept for the
                decoder-layer call convention.
            seq_ctx (SequenceContext | None): Sequence context (unused on this
                path; kept for signature compatibility).

        Returns:
            torch.Tensor: Attention output ``[B, S, D]``.
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        mixed_qkv = torch.cat(
            [self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)],
            dim=-1,
        ).transpose(1, 2)

        # Depthwise causal conv, applied per varlen document so packed
        # sequences do not leak activation across document boundaries; chop
        # the right tail the padding produced.
        bounds = [0, seq_len]
        if seq_ctx is not None and seq_ctx.cu_seq_lens_q is not None:
            bounds = seq_ctx.cu_seq_lens_q.flatten().tolist()
        if len(bounds) > 2:
            out_conv = torch.zeros_like(mixed_qkv)
            for start, end in zip(bounds[:-1], bounds[1:]):
                if end <= start:
                    continue
                seg = mixed_qkv[:, :, start:end]
                y = F.conv1d(
                    seg, self.conv1d.weight, bias=None, padding=self.conv_kernel_dim - 1, groups=self.qkv_dim * 3
                )
                out_conv[:, :, start:end] = y[..., : end - start]
            mixed_qkv = out_conv
        else:
            mixed_qkv = F.conv1d(
                mixed_qkv, self.conv1d.weight, bias=None, padding=self.conv_kernel_dim - 1, groups=self.qkv_dim * 3
            )
            mixed_qkv = mixed_qkv[..., :seq_len]

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2),
            [self.qkv_dim] * 3,
            dim=-1,
        )
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        core_attn_out = get_kda_core_fn()(
            query,
            key,
            value,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
        )

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        # The forget gate runs in fp32 (stability), which makes fla's output fp32;
        # RMSNorm upcasts likewise — cast back before the bf16 o_proj.
        output = self.o_norm(core_attn_out.to(hidden_states.dtype), gate).to(hidden_states.dtype)
        output = output.reshape(batch_size, seq_len, -1)
        return self.o_proj(output)
