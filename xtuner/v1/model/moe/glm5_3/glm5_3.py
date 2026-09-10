# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash (`glm5_next`) text-tower config and model.

Text-only training support for the GLM-5.3-Flash MoE VLM: the vision tower is
not wired yet, so the model consumes the ``text_config`` of the HF checkpoint
and trains the language backbone. Mirrors the DeepSeek-V4 port structure —
the ``MoE`` skeleton drives the forward and the GLM-specific seams (mHC
streams, per-layer KDA/DSA attention, KPool indexer) are the override points.

Reference: ``transformers.models.glm5_next`` and NVIDIA Automodel
``components/models/glm5_next``.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self, override

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.moe import MoE, MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.attention.kda import KdaLinearAttentionConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig, MoEBlock, MoEGate
from xtuner.v1.module.linear import build_linear
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig

from .layers import Glm53DecoderLayer, Glm53FlashSparseAttention


class Glm53HCConfig(BaseModel):
    """mHC hyper-connection parameters (``hc_mult`` / ``hc_eps`` / iters)."""

    model_config = ConfigDict(extra="forbid")

    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20


class Glm53KPoolConfig(BaseModel):
    """KPool indexer parameters of the DSA layers."""

    model_config = ConfigDict(extra="forbid")

    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_kpool: int = 4
    index_kpool_always_select_tail: bool = True


class Glm53FlashConfig(MoEConfig):
    """XTuner configuration for GLM-5.3-Flash's text tower."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Placeholder satisfying the base's computed head-count fields; GLM layers
    # build their own KDA / sparse attention from the fields below.
    attention: MHAConfig | None = None  # type: ignore[assignment]
    layer_types: list[Literal["linear_attention", "deepseek_sparse_attention"]] = []
    hc_cfg: Glm53HCConfig = Glm53HCConfig()
    kpool_cfg: Glm53KPoolConfig = Glm53KPoolConfig()
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256
    linear_attn_num_heads: int = 64
    linear_attn_head_dim: int = 128
    linear_conv_kernel_size: int = 4
    linear_gate_lower_bound: float | None = None
    index_head_dim: int = 128
    index_n_heads: int = 32
    swiglu_limit: float = 10.0

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        """Build the config from the GLM-5.3-Flash HF release directory.

        Args:
            hf_path (str | Path): Path containing ``config.json``.

        Returns:
            Glm53FlashConfig: Config mirroring the HF ``text_config``.
        """
        try:
            cfg = AutoConfig.from_pretrained(hf_path)
            hf_text = cfg.text_config
        except Exception:
            with open(Path(hf_path) / "config.json", encoding="utf-8") as f:
                raw = json.load(f)
            hf_text = SimpleNamespace(**raw.get("text_config", raw))

        layer_types: list[str] = list(getattr(hf_text, "layer_types", []))
        la = getattr(hf_text, "linear_attn_config", None) or SimpleNamespace()

        return cls(
            vocab_size=hf_text.vocab_size,
            max_position_embeddings=hf_text.max_position_embeddings,
            eos_token_id=getattr(hf_text, "eos_token_id", 0),
            pad_token_id=getattr(hf_text, "pad_token_id", None),
            num_hidden_layers=hf_text.num_hidden_layers,
            hidden_size=hf_text.hidden_size,
            intermediate_size=hf_text.intermediate_size,
            rms_norm_eps=hf_text.rms_norm_eps,
            hidden_act=hf_text.hidden_act,
            tie_word_embeddings=hf_text.tie_word_embeddings,
            layer_types=layer_types,
            hc_cfg=Glm53HCConfig(
                hc_mult=hf_text.hc_mult,
                hc_eps=hf_text.hc_eps,
                hc_sinkhorn_iters=hf_text.hc_sinkhorn_iters,
            ),
            kpool_cfg=Glm53KPoolConfig(
                index_topk=hf_text.index_topk,
                index_head_dim=hf_text.index_head_dim,
                index_n_heads=hf_text.index_n_heads,
                index_kpool=hf_text.index_kpool,
                index_kpool_always_select_tail=hf_text.index_kpool_always_select_tail,
            ),
            q_lora_rank=hf_text.q_lora_rank,
            kv_lora_rank=hf_text.kv_lora_rank,
            qk_nope_head_dim=hf_text.qk_nope_head_dim,
            v_head_dim=hf_text.v_head_dim,
            linear_attn_num_heads=getattr(la, "num_heads", 64),
            linear_attn_head_dim=getattr(la, "head_dim", 128),
            linear_conv_kernel_size=getattr(la, "short_conv_kernel_size", 4),
            linear_gate_lower_bound=getattr(la, "gate_lower_bound", None),
            n_routed_experts=hf_text.n_routed_experts,
            n_shared_experts=hf_text.n_shared_experts,
            num_experts_per_tok=hf_text.num_experts_per_tok,
            first_k_dense_replace=hf_text.first_k_dense_replace,
            moe_intermediate_size=hf_text.moe_intermediate_size,
            attention=MHAConfig(
                num_attention_heads=hf_text.num_attention_heads,
                num_key_value_heads=getattr(hf_text, "num_key_value_heads", hf_text.num_attention_heads),
                head_dim=getattr(hf_text, "qk_nope_head_dim", 256),
            ),
            moe_act_fn_cfg=MoEActFnConfig(
                act_type="clamped_swiglu",
                clip_limit=float(getattr(hf_text, "swiglu_limit", 10.0)),
                clamp_shared_expert=True,
            ),
        )

    @property
    def hf_config(self):
        """``None``: no built-in text-only transformers config class exists."""
        return None

    def build(self) -> "Glm53Flash":
        return Glm53Flash(self)


# transformers >= 5.12 is imported lazily to keep module import light.
from transformers import AutoConfig as _AutoConfig  # noqa: E402


AutoConfig = _AutoConfig


class Glm53DenseMLP(nn.Module):
    """Dense FFN with the V4-style clamped SwiGLU.

    Args:
        hidden_size (int): Model hidden size.
        intermediate_size (int): FFN intermediate dimension.
        hidden_act (str): Activation name (``silu``).
        mlp_bias (bool): Whether projections carry biases.
        swiglu_limit (float | None): Clamp bound; ``None`` disables clamping.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mlp_bias: bool = False,
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = build_linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.up_proj = build_linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = build_linear(intermediate_size, hidden_size, bias=mlp_bias)
        self.act_fn = torch.nn.functional.silu if hidden_act == "silu" else None
        self.swiglu_limit = swiglu_limit
        assert self.act_fn is not None, f"Unsupported GLM-5.3 FFN activation: {hidden_act}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm53MoEFFN(nn.Module):
    """MoE FFN: noaux-sigmoid router + shared expert (clamped SwiGLU).

    A thin composition over XTuner's ``MoEGate`` / ``MoEBlock`` so routing,
    bias update, and the expert GEMM stay identical to the standard stack.

    Args:
        cfg (Glm53FlashConfig): Owning model config.
        layer_idx (int): Decoder layer position.
    """

    def __init__(self, *, cfg: "Glm53FlashConfig", layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size_ = cfg.hidden_size
        self.hidden_factor = cfg.hidden_factor
        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.gate = MoEGate(
            hidden_size=cfg.hidden_size,
            n_routed_experts=cfg.n_routed_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            router_config=cfg.router,
            gate_bias=cfg.gate_bias,
            router_compute_dtype=cast(Literal["float32", "native"], cfg.router_compute_dtype),
        )
        self.experts = MoEBlock(
            hidden_size=cfg.hidden_size,
            moe_intermediate_size=cfg.moe_intermediate_size,
            n_routed_experts=cfg.n_routed_experts,
            moe_bias=cfg.moe_bias,
            ep_mesh=None,
            float8_cfg=None,
            moe_act_fn_cfg=cfg.moe_act_fn_cfg,
        )
        self.shared_experts = (
            Glm53DenseMLP(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.moe_intermediate_size * cfg.n_shared_experts,
                hidden_act=cfg.hidden_act,
                mlp_bias=cfg.mlp_bias,
                swiglu_limit=cfg.moe_act_fn_cfg.clip_limit if cfg.moe_act_fn_cfg.clamp_shared_expert else None,
            )
            if cfg.n_shared_experts > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route, run experts, and combine routed + shared outputs.

        Args:
            x (torch.Tensor): ``[B, S, D]`` sub-layer input.

        Returns:
            torch.Tensor: FFN output ``[B, S, D]``.
        """
        orig_shape = x.shape
        flat = x.view(-1, x.shape[-1])
        router_results = self.gate(x)
        topk_ids = router_results["topk_ids"]
        topk_weights = router_results["topk_weights"]
        self._last_router = (router_results["logits"], topk_weights, topk_ids)

        n_tokens, k = topk_ids.shape
        flat_expert = self.experts.fused_w1w3.weight  # [E * 2*I, D]
        flat_down = self.experts.fused_w2.weight  # [E * D, I]
        # Grouped execution per selected expert token (ep_size == 1):
        out = torch.zeros_like(flat)
        token_index = torch.arange(n_tokens, device=flat.device).repeat_interleave(k)
        flat_tokens = flat[token_index]
        sel_ids = topk_ids.reshape(-1)
        sel_w = topk_weights.reshape(-1)
        for expert in range(sel_ids.max().item() + 1):
            mask = sel_ids == expert
            if not mask.any():
                continue
            tokens = flat_tokens[mask]
            two_i = 2 * self.experts.intermediate_size
            w1w3 = flat_expert[expert * two_i : (expert + 1) * two_i]
            w2 = flat_down[expert * self.hidden_size_ : (expert + 1) * self.hidden_size_]  # [D, I]
            h = tokens @ w1w3.T
            act_out = self.experts.moe_act(h, split_dim=-1)
            proj = act_out @ w2.T
            out.index_add_(0, token_index[mask], (proj * sel_w[mask, None]).to(out.dtype))
        moe_out = out.view(*orig_shape)
        if self.shared_experts is not None:
            moe_out = moe_out + self.shared_experts(x)
        return moe_out


class Glm53Flash(MoE):
    """GLM-5.3-Flash text tower (mHC + KDA/DSA hybrid + MoE)."""

    config: Glm53FlashConfig

    def __init__(self, config: Glm53FlashConfig) -> None:
        self._hc_mult = config.hc_cfg.hc_mult
        super().__init__(config)

    @override
    def build_embeddings(self, config: MoEConfig):
        return nn.Embedding(config.vocab_size, config.hidden_size)

    @override
    def build_layers(self, config: MoEConfig) -> nn.ModuleDict:
        glm_cfg = cast(Glm53FlashConfig, config)
        layers: dict[str, nn.Module] = {}
        for layer_idx in range(glm_cfg.num_hidden_layers):
            layer_type = glm_cfg.layer_types[layer_idx]
            is_moe = layer_idx >= glm_cfg.first_k_dense_replace
            if layer_type == "linear_attention":
                self_attn = KdaLinearAttentionConfig(
                    num_heads=glm_cfg.linear_attn_num_heads,
                    head_dim=glm_cfg.linear_attn_head_dim,
                    conv_kernel_dim=glm_cfg.linear_conv_kernel_size,
                    gate_lower_bound=glm_cfg.linear_gate_lower_bound,
                    rms_norm_eps=glm_cfg.rms_norm_eps,
                ).build(hidden_size=glm_cfg.hidden_size, layer_idx=layer_idx)
            else:
                self_attn = Glm53FlashSparseAttention(
                    hidden_size=glm_cfg.hidden_size,
                    num_attention_heads=glm_cfg.num_attention_heads,
                    q_lora_rank=glm_cfg.q_lora_rank,
                    kv_lora_rank=glm_cfg.kv_lora_rank,
                    qk_nope_head_dim=glm_cfg.qk_nope_head_dim,
                    v_head_dim=glm_cfg.v_head_dim,
                    index_head_dim=glm_cfg.kpool_cfg.index_head_dim,
                    index_n_heads=glm_cfg.kpool_cfg.index_n_heads,
                    index_topk=glm_cfg.kpool_cfg.index_topk,
                    index_kpool=glm_cfg.kpool_cfg.index_kpool,
                    index_kpool_always_select_tail=glm_cfg.kpool_cfg.index_kpool_always_select_tail,
                )
            if is_moe:
                mlp = Glm53MoEFFN(cfg=glm_cfg, layer_idx=layer_idx)
            else:
                mlp = Glm53DenseMLP(
                    hidden_size=glm_cfg.hidden_size,
                    intermediate_size=glm_cfg.intermediate_size,
                    hidden_act=glm_cfg.hidden_act,
                    mlp_bias=glm_cfg.mlp_bias,
                    swiglu_limit=glm_cfg.swiglu_limit,
                )
            layers[str(layer_idx)] = Glm53DecoderLayer(
                layer_idx=layer_idx,
                layer_type=layer_type,
                is_moe=is_moe,
                hidden_size=glm_cfg.hidden_size,
                hc_mult=glm_cfg.hc_cfg.hc_mult,
                hc_eps=glm_cfg.hc_cfg.hc_eps,
                hc_sinkhorn_iters=glm_cfg.hc_cfg.hc_sinkhorn_iters,
                rms_norm_eps=glm_cfg.rms_norm_eps,
                self_attn=self_attn,
                mlp=mlp,
            )
        return nn.ModuleDict(layers)

    @override
    def build_mtp_block(self, config: MoEConfig):
        # The GLM-5.3-Flash release checkpoint ships no MTP weights.
        return None

    @override
    def build_rotary_embedding(self, config):
        # GLM-5.3-Flash is rope-free: KDA needs no rope and the DSA layers run
        # with ``qk_rope_head_dim == 0`` (nope channels only). Return an
        # identity callable so the base ``_forward``'s rotary call stays
        # harmless and the (cos, sin) pair flows through as ``None``.
        def _identity_rotary(hidden_states, position_ids):  # noqa: ANN001
            return (None, None)

        return _identity_rotary

    @override
    def _decoder_stack(
        self,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        output: dict,
        keep_router: bool,
        balancing_ctx,
        z_ctx,
        nonpad_indices,
        non_pad_token: int,
        num_tokens_global,
        z_world_size: int,
    ) -> torch.Tensor:
        """mHC decoder stack: expand streams, run layers, return streams."""
        del position_embeddings  # rope-free model
        hc_mult = self.config.hc_cfg.hc_mult
        hidden_streams = (
            hidden_states.unsqueeze(-2).expand(-1, -1, hc_mult, -1).contiguous()
            if hidden_states.dim() == 3
            else hidden_states
        )

        for idx, decoder_layer in self.layers.items():
            layer_idx = int(idx)
            hidden_streams = decoder_layer(hidden_streams, (None, None), seq_ctx)
            hidden = hidden_streams.mean(dim=2)
            if layer_idx >= self.config.first_k_dense_replace and self._should_compute_aux_loss(layer_idx):
                ffn = decoder_layer.mlp
                router_logits, router_weights, router_topk_ids = ffn._last_router
                hidden_streams = hidden_streams.clone()
                hidden_streams[..., 0, :] = self.aux_loss.accumulate(
                    selected_router_weights=router_weights.index_select(0, nonpad_indices).contiguous().float(),
                    selected_router_logits=router_logits.index_select(0, nonpad_indices).contiguous().float(),
                    selected_experts=router_topk_ids.index_select(0, nonpad_indices).contiguous(),
                    hidden_states=hidden_streams[..., 0, :],
                    balancing_ctx=balancing_ctx,
                    z_ctx=z_ctx,
                    num_tokens_local=non_pad_token,
                    num_tokens_global=num_tokens_global,
                    world_size=z_world_size,
                )
            if self.config.return_hidden_states:
                output["hidden_states"].append(hidden)

        # Collapse the mHC streams back to [B, S, D] before the final norm
        # (GLM's HyperHead: unweighted mean over streams).
        return hidden_streams.mean(dim=2)

    @override
    def _should_compute_aux_loss(self, layer_idx: int) -> bool:
        # GLM-5.3's noaux sigmoid router emits score-compatible stats on every
        # sparse layer; dense prefix layers never reach this gate.
        return True


class Glm53FlashTowerConfig(Glm53FlashConfig):
    """Released GLM-5.3-Flash text-tower dimensions (320B total / 18B active)."""

    router: NoAuxRouterConfig = NoAuxRouterConfig(
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        router_scaling_factor=2.5,
    )
    vocab_size: int = 154880
    max_position_embeddings: int = 1048576
    eos_token_id: list[int] | int = [154820, 154827, 154829]
    num_hidden_layers: int = 45
    hidden_size: int = 4096
    intermediate_size: int = 12288
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    moe_intermediate_size: int = 2048
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256

    def build(self) -> Glm53Flash:
        return Glm53Flash(self)
