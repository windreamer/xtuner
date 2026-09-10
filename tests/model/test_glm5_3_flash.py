# Copyright (c) OpenMMLab. All rights reserved.
"""Random-small-model tests for the GLM-5.3-Flash text tower.

These exercise the full forward → loss → backward path through the hybrid
KDA / KPool-DSA / MoE stack with mHC residual streams, plus config
derivations. Pure random weights (``init_weights``) — no checkpoint needed.
"""

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.glm5_3 import Glm53FlashConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEActFnConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig


_VOCAB = 128
_SEQ = 64


def _build_config(layer_types=None, num_layers=4):
    return Glm53FlashConfig(
        vocab_size=_VOCAB,
        max_position_embeddings=2048,
        eos_token_id=0,
        num_hidden_layers=num_layers,
        hidden_size=64,
        intermediate_size=128,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=4, num_key_value_heads=4, head_dim=16),
        layer_types=layer_types
        or ["linear_attention", "linear_attention", "deepseek_sparse_attention", "linear_attention"],
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_intermediate_size=32,
        router=NoAuxRouterConfig(
            n_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            norm_topk_prob=True,
            router_scaling_factor=2.5,
        ),
        moe_act_fn_cfg=MoEActFnConfig(act_type="clamped_swiglu", clip_limit=10.0, clamp_shared_expert=True),
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        linear_attn_num_heads=2,
        linear_attn_head_dim=16,
        linear_conv_kernel_size=4,
        linear_gate_lower_bound=-5.0,
        swiglu_limit=10.0,
    )


def _run_forward_backward(model, seed=0):
    torch.manual_seed(seed)
    input_ids = torch.randint(0, _VOCAB // 2, (1, _SEQ), device="cuda")
    seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))
    loss_cfg = CELossConfig(mode="eager")
    loss_ctx = loss_cfg.build(data={"shifted_labels": input_ids.clone()}, sp_mesh=None)
    loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
    output = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
    output["loss"].backward()
    return output["loss"]


@pytest.mark.gpu
class TestGlm53FlashRandomSmall:
    """Whole-tower forward/backward on random weights, mixed KDA/DSA stack."""

    def test_forward_backward_mixed_stack(self):
        torch.manual_seed(0)
        model = _build_config().build().cuda().to(torch.bfloat16)
        model.init_weights()
        loss = _run_forward_backward(model, seed=1)
        assert loss.item() == loss.item(), "loss is NaN"
        assert loss.item() > 0
        grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        assert grad_sum > 0, "backward produced no gradients"

    def test_kda_only_stack(self):
        torch.manual_seed(0)
        model = _build_config(layer_types=["linear_attention"] * 4).build().cuda().to(torch.bfloat16)
        model.init_weights()
        loss = _run_forward_backward(model, seed=2)
        assert loss.item() == loss.item()
        grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        assert grad_sum > 0

    def test_dsa_only_stack(self):
        torch.manual_seed(0)
        model = _build_config(layer_types=["deepseek_sparse_attention"] * 4).build().cuda().to(torch.bfloat16)
        model.init_weights()
        loss = _run_forward_backward(model, seed=3)
        assert loss.item() == loss.item()
        grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        assert grad_sum > 0

    def test_repeat_runs_are_finite(self):
        """Three different seeds all produce finite losses (no amortized NaN)."""
        torch.manual_seed(0)
        model = _build_config().build().cuda().to(torch.bfloat16)
        model.init_weights()
        for seed in range(3):
            model.zero_grad(set_to_none=True)
            loss = _run_forward_backward(model, seed=seed)
            assert loss.item() == loss.item(), f"seed {seed}: loss is NaN"
