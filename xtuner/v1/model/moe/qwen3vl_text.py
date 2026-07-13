import re

import torch

from .qwen3 import Qwen3MoE, Qwen3MoE30BA3Config, Qwen3MoE235BA22Config


class Qwen3VLTextMoE(Qwen3MoE):
    def to_hf_key_list(self, key: str) -> list[str]:
        if "layers" in key or "embed_tokens" in key:
            key = "model.language_model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate)", r"layers.\1.mlp.\2", key)

        if "fused_w1w3.weight" in key:
            key = key.replace("fused_w1w3.weight", "gate_up_proj")
        elif "fused_w2.weight" in key:
            key = key.replace("fused_w2.weight", "down_proj")
        if "fused_w1w3.bias" in key:
            key = key.replace("fused_w1w3.bias", "gate_up_proj_bias")
        elif "fused_w2.bias" in key:
            key = key.replace("fused_w2.bias", "down_proj_bias")

        if key.startswith("norm."):
            return [key.replace("norm.", "model.language_model.norm.")]
        elif key.startswith("rotary_emb."):
            # FoPE has model.rotary_emb.sin_coef and model.rotary_emb.cos_coef in the safetensors
            return [key.replace("rotary_emb.", "model.language_model.rotary_emb.")]
        else:
            return [key]

    def hf_tensor_to_canonical(self, name: str, loaded_tensor: torch.Tensor) -> torch.Tensor:
        if "fused_w1w3.weight" in name:
            # hf: num_experts, hidden_size, 2 * expert_dim
            # xtuner: num_experts * 2 * expert_dim, hidden_size
            num_experts, hidden_size = loaded_tensor.shape[:2]
            loaded_tensor = loaded_tensor.transpose(1, 2)  # num_experts, 2 * expert_dim, hidden_size
            # num_experts * 2 * expert_dim, hidden_size
            loaded_tensor = loaded_tensor.reshape(-1, hidden_size)

        elif "fused_w2.weight" in name:
            # hf: num_experts, expert_dim, hidden_size
            # xtuner: num_experts * hidden_size, expert_dim
            loaded_tensor = loaded_tensor.transpose(1, 2).flatten(0, 1)

        return loaded_tensor

    def param_to_safetensor(
        self,
        safetensor: torch.Tensor,
        hf_param_name: str,
    ):
        assert isinstance(hf_param_name, str)
        if "gate_up_proj" in hf_param_name:
            # xtuner: num_experts * 2 * expert_dim, hidden_size
            # hf: num_experts, hidden_size, 2 * expert_dim
            num_experts = self.config.n_routed_experts
            hidden_size = safetensor.size(1)
            safetensor = safetensor.reshape(num_experts, -1, hidden_size)  # num_experts, 2 * expert_dim, hidden_size
            safetensor = safetensor.transpose(1, 2).contiguous()  # num_experts, hidden_size, 2 * expert_dim
        elif "down_proj" in hf_param_name:
            # xtuner: num_experts * hidden_size, expert_dim
            # hf: num_experts, expert_dim, hidden_size
            num_experts = self.config.n_routed_experts
            expert_dim = safetensor.size(1)
            safetensor = safetensor.reshape(num_experts, -1, expert_dim).transpose(1, 2).contiguous()
        return safetensor

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def _post_layer(self, hidden_states, idx, seq_ctx):
        # Inject the matching deepstack visual embeds after each of the first
        # ``len(deepstack_visual_embeds)`` layers; text-only batches (no visual
        # embeds) fall through unchanged. Everything else in the forward — embed,
        # rope, the offload window, the final norm/aux finalize — uses the MoE base.
        deepstack_visual_embeds = seq_ctx.deepstack_visual_embeds
        if deepstack_visual_embeds is not None and int(idx) in range(len(deepstack_visual_embeds)):
            assert seq_ctx.visual_pos_masks is not None
            hidden_states = self._deepstack_process(
                hidden_states, seq_ctx.visual_pos_masks, deepstack_visual_embeds[int(idx)]
            )
        return hidden_states


class Qwen3VLTextMoE30BA3Config(Qwen3MoE30BA3Config):
    def build(self) -> Qwen3MoE:
        return Qwen3VLTextMoE(self)


class Qwen3VLTextMoE235BA22Config(Qwen3MoE235BA22Config):
    def build(self) -> Qwen3MoE:
        return Qwen3VLTextMoE(self)
