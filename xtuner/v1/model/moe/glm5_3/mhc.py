# Copyright (c) OpenMMLab. All rights reserved.
"""Manifold-constrained Hyper-Connections (mHC) for GLM-5.3-Flash.

Port of ``transformers.models.glm5_next.Glm5NextTextHyperConnection`` /
``Glm5NextTextHyperHead``. The split-Sinkhorn math is identical to
DeepSeek-V4's :func:`hc_split_sinkhorn`; this module keeps GLM's parameter
layout — two flat per-site groups ``(hc_attn_*, hc_ffn_*)`` per layer, each a
``fn`` projection plus ``base`` / ``scale`` vectors — and GLM's final stream
collapse, an **unweighted mean** over the ``hc_mult`` streams (V4 uses a
learned sigmoid-gate reduce).

All math runs in fp32 internally (bf16-stable across the 20 Sinkhorn
iterations) and casts back to the stream dtype.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mhc_split_sinkhorn(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``mixes`` into (pre, post, comb) and Sinkhorn-project ``comb``.

    Args:
        mixes (torch.Tensor): ``[..., (2 + hc_mult) * hc_mult]`` mix logits (fp32).
        scale (torch.Tensor): ``[3]`` per-group learned scales.
        base (torch.Tensor): ``[(2 + hc_mult) * hc_mult]`` per-slot bias.
        hc_mult (int): Stream count ``H``.
        iters (int): Sinkhorn-Knopp iterations for ``comb``.
        eps (float): Stabilizer for ``pre`` and the Sinkhorn normalizers.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(pre, post, comb)``.
    """
    pre_w, post_w, comb_w = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    pre_b, post_b, comb_b = base.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    pre_scale, post_scale, comb_scale = scale.unbind(-1)

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc_mult, hc_mult) * comb_scale + comb_b.view(hc_mult, hc_mult)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class GlmMHC(nn.Module):
    """One mHC site (attention or ffn) of a GLM-5.3-Flash decoder layer.

    Args:
        hidden_size (int): Model hidden size.
        hc_mult (int): Number of residual streams.
        hc_eps (float): Sinkhorn / pre stabilizer.
        hc_sinkhorn_iters (int): Sinkhorn iteration count.
        rms_norm_eps (float): Epsilon for the unweighted input RMSNorm.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        hc_eps: float,
        hc_sinkhorn_iters: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.input_norm = nn.RMSNorm(hc_mult * hidden_size, eps=rms_norm_eps, elementwise_affine=False)
        mix = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(torch.empty(mix, hc_mult * hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def custom_init_weights(self) -> set[str]:
        """HF-matching init: normal ``fn``, zero ``base``, ones ``scale``."""
        torch.nn.init.normal_(self.fn, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.base)
        torch.nn.init.ones_(self.scale)
        return {"fn", "base", "scale"}

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix the incoming streams and collapse them for the sub-layer.

        Args:
            hidden_streams (torch.Tensor): ``[B, S, H, D]`` residual streams.


        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(collapsed,
            post, comb)`` — ``collapsed`` is the ``[B, S, D]`` sub-layer
            input; ``post`` ``[B, S, H]`` and ``comb`` ``[B, S, H, H]`` are
            applied to the sub-layer output by the caller.
        """
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())  # [B, S, H*D]
        mixes = F.linear(flat, self.fn.float())
        pre, post, comb = mhc_split_sinkhorn(
            mixes, self.scale.float(), self.base.float(), hc, self.hc_sinkhorn_iters, self.hc_eps
        )
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return collapsed, post, comb


class GlmMHCHead(nn.Module):
    """Final GLM-5.3-Flash stream collapse: unweighted mean over streams."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        """Collapse ``[B, S, H, D]`` streams to ``[B, S, D]`` by mean."""
        return hidden_streams.mean(dim=2)
