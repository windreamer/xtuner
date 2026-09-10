# Copyright (c) OpenMMLab. All rights reserved.
"""Op-selection seam for the KDA recurrence kernel.

Mirrors ``xtuner.v1.ops.attn_imp``: production runs the fla kernel when it is
importable, parity tests force the HF-exact eager fallback via
``XTUNER_HF_IMPL``. The selector reads the env var live so a test can toggle
it around a single model build.

The torch fallback is a line-for-line port of transformers'
``Glm5NextText`` ``chunk_kimi_delta_attention`` (the
``use_kernel_func_from_hub_with_fallback`` body), so ``XTUNER_HF_IMPL``
reproduces HF numerics exactly.
"""

import os

import torch
import torch.nn.functional as F

from xtuner.v1.utils import get_logger


logger = get_logger()

FALLBACK_CHUNK_SIZE = 64

try:
    from fla.ops.kda import chunk_kda
except ImportError:  # pragma: no cover - fla is an optional training dependency
    chunk_kda = None


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    # Aligns with fla's l2norm: sqrt(sum(x^2)) + eps, division instead of max().
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def _chunk_kda_torch_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Chunked KDA in fp32 — port of transformers' ``chunk_kimi_delta_attention``.

    Args:
        query (torch.Tensor): ``[B, T, H, K]``.
        key (torch.Tensor): ``[B, T, H, K]``.
        value (torch.Tensor): ``[B, T, H, V]``.
        g (torch.Tensor): Log-decay ``[B, T, H]``.
        beta (torch.Tensor): ``[B, T, H]``, pre-sigmoid gate is applied inside.
        use_qk_l2norm_in_kernel (bool): L2-normalise q/k in fp32 first.

    Returns:
        torch.Tensor: Core attention output ``[B, T, H, V]`` in the input dtype.
    """
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)

    initial_dtype = query.dtype
    chunk_size = FALLBACK_CHUNK_SIZE

    # fla computes the whole recurrence in fp32.
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)]

    batch_size, num_heads, seq_len, _k_head_dim = key.shape
    scale = 1 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    total_len = seq_len + pad_size

    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, g, k_beta, v_beta)
    ]

    # Intra-chunk attention with cumulative decay.
    g = g.cumsum(dim=-2)
    diag_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(diag_mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = torch.zeros(
        batch_size, num_heads, query.shape[-1], value.shape[-1], dtype=value.dtype, device=value.device
    )
    core_attn_out = torch.zeros_like(value)

    strict_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(total_len // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        # Inter chunk: state carries the pre-chunk summary, decayed to now.
        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        # Intra chunk: causal attention within the chunk with decay.
        attn_intra = (
            (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1).masked_fill(strict_upper, 0)
        )
        # KDA update rule: the chunk's key contribution is discounted by the
        # decay-weighted state ("v_new = v_i - k_cumdecay @ state").
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :seq_len]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)


def get_kda_core_fn():
    """Return the KDA chunk kernel selected by ``XTUNER_HF_IMPL``.

    Production: fla ``chunk_kda`` when fla is installed, else the torch
    fallback (with a warning). Parity: always the torch fallback so numerics
    match HF exactly.

    Returns:
        Callable: ``fn(query, key, value, g, beta, use_qk_l2norm_in_kernel)``
        returning the core attention output tensor.
    """
    use_hf_impl = os.getenv("XTUNER_HF_IMPL", "0") == "1"
    if use_hf_impl or chunk_kda is None:
        if chunk_kda is None and not use_hf_impl:
            logger.log_once("fla is not installed; KDA runs the torch fallback (slow).")
        return _chunk_kda_torch_fallback

    def fla_chunk(query, key, value, g, beta, use_qk_l2norm_in_kernel=True):
        # fla chunk_kda contract: ``g`` is per-channel ``[B, T, H, K]`` log-decay
        # (the forget gate already emits that shape); ``beta`` is the raw logit
        # and the kernel fuses its sigmoid.
        out, _ = chunk_kda(
            query,
            key,
            value,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=True,
            output_final_state=False,
        )
        return out

    return fla_chunk
