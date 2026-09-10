# GLM-5.3-Flash (glm5_next) integration design

## 1. What the model is

`ZhipuAI/GLM-5.3-Flash` — `model_type: "glm5_next"`, architecture
`Glm5NextForConditionalGeneration` (transformers >= 5.16 ships it built-in; no
`trust_remote_code` needed). A **vision-language MoE** whose text tower is a
**hybrid of KDA linear attention and DeepSeek Sparse Attention (MLA)**, with
**Manifold-constrained Hyper-Connections (mHC)** on every layer.

Reference HF implementation: `transformers/models/glm5_next/modeling_glm5_next.py`
(2385 lines). Weights live on ModelScope under `ZhipuAI/GLM-5.3-Flash` (62
shards, bf16; fp8 release is a separate repo).

| Tower | Shape |
|-------|-------|
| Vision (`model.visual.*`) | GLM-5.3 vision: `patch_embed`, `blocks.{N}` (attn qkv/proj + q/k_norm, mlp), `norm1/2`, `patch_merger.proj + post_projection_norm + gate/up/down` (GELU + **clamped** SwiGLU). depth 24, hidden 1024, `out_hidden_size` 4096, `spatial_merge_size` 2 |
| Text (`model.language_model.*`) | MoE, 45 layers, hidden 4096, `first_k_dense_replace=3`, `hc_mult=4` (mHC on **every** layer), vocab 154880 |

### Text layer schedule (from `layer_types`)

- **34 × `linear_attention`** — Kimi-style KDA (Kimi Delta Attention).
  `self_attn` weights: `q/k/v_proj`, per-axis short conv (`q_conv1d`,
  `k_conv1d`, `v_conv1d`, kernel 4), `b_proj` (beta), forget gate
  (`f_a_proj`, `f_b_proj`, `dt_bias`, `A_log`, `gate_lower_bound=-5.0` safe
  sigmoid decay), `g_a_proj`/`g_b_proj` gate, gated RMSNorm (`o_norm`),
  `o_proj`. Kernel: `chunk_kda` / `fused_recurrent_kda` (fla; HF ships a
  torch fallback via `use_kernel_func_from_hub_with_fallback`).
- **11 × `deepseek_sparse_attention`** — MLA (q_lora 1536, kv_lora 512,
  qk_nope 256, v_head 256, **qk_rope_head_dim 0** — no decoupled rope on the
  KV side) + a DSA indexer with **k-pool compression** (`index_kpool=4`,
  `index_kpool_always_select_tail=True`, `index_topk=2048`).
  `indexer_types` is all `"full"` → every DSA layer computes its own top-k
  (no cross-layer sharing — glm52's `skip_topk` path is unused here).

### MoE

`n_routed_experts=288`, `num_experts_per_tok=8`, `n_shared_experts=1`,
`first_k_dense_replace=3` (dense MLP on layers 0–2), sigmoid scoring,
`noaux_tc` (`n_group=1, topk_group=1` → same spelling as V4: **no group
limitation**), `norm_topk_prob=True`, `routed_scaling_factor=2.5`,
`moe_intermediate_size=2048`, router in fp32.

### mHC hyper-connections

`hc_mult=4` on **all 45 layers** (V4 differs: same mechanism but the HC head
is a sigmoid-gated reduce). GLM variant:

- `Glm5NextTextHyperConnection` owns `hc_{attn,ffn}_{base,fn,scale}`:
  `fn` is `[3H, H*D]` (pre / post / comb logits), `base` `[3H]`, `scale`
  `[3]`; per-site `input_norm` (unweighted RMSNorm) then
  `sigmoid(·)+eps` pre / `2·sigmoid(·)` post / `softmax+eps` → Sinkhorn(20)
  comb, same as V4's `hc_split_sinkhorn`.
- `Glm5NextTextHyperHead` (final collapse): **unweighted mean** over the 4
  streams — V4 uses a learned sigmoid-gate reduce instead.
- Weight layout: `layers.{i}.hc_attn_{base,fn,scale}`, `hc_ffn_{...}` — flat
  per-layer params, not nested modules.

### Other text facts

- `swiglu_limit=10.0` clamp on routed/shared/visible MLPs (same clamped
  SwiGLU family as V4; vision patch-merger also clamps).
- `num_nextn_predict_layers=1` — but **no `model.layers.45` / `mtp.*` keys in
  the checkpoint** (0 matches) → MTP is config-only, nothing to load. Defer
  MTP wiring (same decision as the V4 port's first pass).
- RoPE: plain global theta on full-attention layers (no partial rotary, no
  mrope on the text tower); `max_position_embeddings=1048576`.
- `tie_word_embeddings=false`; `lm_head.weight` present.

## 2. Reuse map — what already exists

| Need | Existing XTuner piece | Delta |
|---|---|---|
| MLA + DSA indexer core | `xtuner/v1/module/attention/dsa/` (V4) and glm52 `DSAMultiLatentAttention` | k-pool compression is new; `qk_rope_head_dim=0` (rope-free KV) is a config point, code already branches |
| Cross-layer top-k sharing | glm52 `dsa_topk_sharing` + `previous_layer_results` seam | unused here (`indexer_types` all `"full"`) — the seam still carries the plumbing |
| Hyper-connections | `xtuner/v1/module/decoder_layer/deepseek_v4/hc_block.py`, `hc_sinkhorn.py` | mHC mapping matches (sigmoid/2σ/softmax+Sinkhorn); the **head** differs (unweighted mean vs learned gate) and params are flat `hc_*_{base,fn,scale}` per site → add a GLM variant beside the V4 one |
| Linear attention | `GatedDeltaNetConfig` (`module/attention/gated_deltanet.py`, fla `chunk_gated_delta_rule`) | KDA is a **different recurrence** (per-head decay gate + data-dependent beta, fla `chunk_kda`) → new module `KdaLinearAttentionConfig`, ops under `xtuner/v1/ops/kda/` wrapping fla with a torch fallback mirroring HF's |
| MoE (noaux sigmoid, shared expert, clamped swiglu) | `MoE` + `NoAuxRouterConfig` (`n_group=1/topk_group=1`), `MoEActFnConfig(clamped_swiglu)` | none — exact V4 spelling |
| Hybrid layer schedule | `layer_types`-driven `build_layers` (Dense side has the linear/full pattern; V4 has the dsa side) | new text model joins the two: KDA vs DSA per `layer_types` |
| Vision tower | qwen3_vl / qwen3_5 vision configs are structurally close but the **patch merger differs** (GELU-then-clamped-SwiGLU, `post_projection_norm`) | new `Glm5NextVisionConfig`-equivalent + modeling port; block/qkv parts are reusable patterns |
| Compose base | `BaseComposeConfig` / `BaseComposeModel` (`model.language_model` prefix matches this checkpoint exactly) | subclass with the three towers |
| fp8 | `weight_scale_inv` suffixes throughout → fp8 blockwise release | BF16 first; fp8 later via existing `float8` plumbing |

## 3. File layout

```
xtuner/v1/module/attention/kda/            # KDA linear attention (config + module + ops shim)
xtuner/v1/model/moe/glm5_3/                # text model + config + vision + projector
  glm5_3_text.py                           # Glm53FlashText(MoE) — hybrid KDA/DSA stack
  glm5_3_config.py                         # Glm53FlashConfig (BaseComposeConfig subclass)
  glm5_3_vision.py                         # vision tower + patch merger port
tests/model/test_glm5_3_flash_config.py    # from_hf derivation regressions
tests/model/test_glm5_3_flash.py           # parity + round-trip (GPU)
ci/config/glm5_3_flash.py                  # training smoke config
docs/design/model/glm_5_3_flash.md         # this file
```

## 4. Sequencing

1. **Config + registration** (`from_hf` reading the 3-level HF config:
   `text_config` + `vision_config`), `to_hf_key_list` for the
   `model.language_model` prefix and the fused-expert 1→288 mapping.
2. **KDA module** (fla `chunk_kda` primary; HF torch fallback for parity).
3. **DSA k-pool indexer extension** (behind a config flag on the existing
   indexer; V4 path untouched).
4. **mHC variant + text model assembly** (`V4DecoderLayer`-style layer class:
   two mHC sites per layer, KDA or DSA+MLA attention site, MoE/visible MLP).
5. **Vision tower + compose wiring**; parity + round-trip tests.
6. **MTP wiring** — deferred: checkpoint has no MTP weights despite
   `num_nextn_predict_layers=1`; revisit when a release ships them.

## 5. Open questions (blocked on release inspection)

- `index_share_for_mtp_iteration=true` semantics when MTP ships.
- Exact `image_token_id` handling in the compose forward (154854) vs
  qwen3_vl's convention.
