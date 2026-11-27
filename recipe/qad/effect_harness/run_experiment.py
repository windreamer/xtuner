# T9 sim: one QAD experiment (PTQ baseline -> micro-run -> export -> replay -> gates B & C).
#
# Pipeline per experiment (design doc §10.2, §9.2):
#   1. holdout KL of the fake-quant student at the INITIAL weights = PTQ baseline (amax scales,
#      no training).
#   2. QADTrainer.fit() micro-run with holdout-KL evaluation hooks (KL curve).
#   3. holdout KL at the end of training in the scale-freeze phase = KL(B_end).
#   4. export_qad(master weights + frozen scales) -> fresh model -> load_qad_export replay on
#      the SAME holdout -> KL(C); gate C: KL(C) == KL(B_end) (+ torch.equal on logits).
#   5. Gate A anchor: the teacher IS the initial-weight BF16 copy, so KL(A) == 0 by construction;
#      verified numerically once (bf16 student vs teacher on the holdout).

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets.collator import build_text_ctx_labels
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.sft_tokenize_fn import OpenaiTokenizeFunctionConfig
from xtuner.v1.model import Qwen3DenseConfig, get_model_config_from_hf
from xtuner.v1.quant.config import KVQConfig, QADQuantConfig
from xtuner.v1.quant.export import export_qad, load_qad_export
from xtuner.v1.quant.fake_quant_linear import FakeQuantLinear, apply_quantization
from xtuner.v1.quant.kv_cache import KVQuantHandle, apply_kv_quantization
from xtuner.v1.quant.loss import QADLossConfig
from xtuner.v1.quant.teacher import QADTeacherAdapterConfig
from xtuner.v1.rl.distillation.config import TrainTeacherConfig
from xtuner.v1.train.qad_trainer import QADTrainer, QADTrainerConfig, _attention_modules, _loss_mask
from xtuner.v1.train.trainer import HooksConfig, TrainerConfig
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


def _model_cfg_compile_off(model_cfg: Any) -> Any:
    """Normalize a loaded model config for the sim: compile off + qk-norm off.

    ``Qwen3DenseConfig.from_hf`` hardcodes ``qk_norm=True`` (Qwen3 convention); the real GLM
    text config has no qk-norm, and the ``self_attn.q_norm``/``k_norm`` RMSNorm submodules it
    creates are false-positive attention matches for ``load_qad_export``'s full-model KV walk
    (RMSNorm exposes no k_proj). Applied identically to student, teacher, clone and replay so
    every side of the KL chain shares one architecture.
    """
    model_cfg.compile_cfg = False
    if hasattr(model_cfg, "attention") and getattr(model_cfg.attention, "qk_norm", False):
        model_cfg.attention.qk_norm = False
    return model_cfg


def _load_model_cfg(model_dir: str) -> Any:
    return _model_cfg_compile_off(get_model_config_from_hf(Path(model_dir)))


def _full(t: torch.Tensor | Any) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t.detach().clone()


def build_holdout_batches(
    holdout_jsonl: Path,
    tokenizer: Any,
    pack_max_length: int,
    pad_token_id: int,
    max_rows: int,
) -> list[list[dict[str, Any]]]:
    """Tokenize the holdout jsonl and pack it exactly like the training collator."""
    tokenize_fn = OpenaiTokenizeFunctionConfig(chat_template="intern-s1", max_length=pack_max_length).build(
        tokenizer=tokenizer
    )
    items: list[dict[str, Any]] = []
    with holdout_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(tokenize_fn(json.loads(line)))
    # Greedy whole-sample packing into rows of <= pack_max_length tokens.
    rows: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_len = 0
    for item in items:
        n = int(item["num_tokens"])
        if n > pack_max_length:
            continue
        if cur_len + n > pack_max_length:
            rows.append(cur)
            cur, cur_len = [], 0
        cur.append(item)
        cur_len += n
        if len(rows) >= max_rows:
            break
    if cur and len(rows) < max_rows:
        rows.append(cur)
    batches = []
    for row in rows:
        seq_ctx, shifted_labels, _ = build_text_ctx_labels(
            row, pack_max_length, pad_token_id, pack_to_max_length=True, pad_chunk_size=256
        )
        batches.append([{"seq_ctx": seq_ctx, "shifted_labels": shifted_labels}])
    return batches


def holdout_kl_for_model(
    trainer: QADTrainer,
    model: Any,
    holdout_batches: list[list[dict[str, Any]]],
    sp_mesh: Any = None,
) -> dict[str, float]:
    """Masked mean top-K forward KL of ``model`` vs the trainer's frozen teacher.

    Mirrors QADTrainer.evaluate_holdout (same teacher targets, same loss context, same
    reduction) but accepts any model, so the exported/replayed model is scored on exactly the
    same metric as the in-training student.
    """
    loss_cfg = trainer._qad_loss_cfg()
    kl_sum = 0.0
    token_sum = 0.0
    with torch.no_grad():
        for data_batch in holdout_batches:
            for item in data_batch:
                targets = trainer._teacher_targets(item["seq_ctx"], loss_cfg)
                item["teacher_logprobs"] = targets["teacher_logprobs"]
                if "target_token_ids" in targets:
                    item["target_token_ids"] = targets["target_token_ids"]
                item["loss_mask"] = _loss_mask(item, loss_cfg)
            ctx_list = model.build_loss_ctx_batch(data_batch, sp_mesh=sp_mesh)
            for item, ctx in zip(data_batch, ctx_list):
                output = model(seq_ctx=item["seq_ctx"].to(DEVICE), loss_ctx={"lm": ctx["lm"]})
                extra = getattr(output, "extra_info", None)
                if not isinstance(extra, dict):
                    raise RuntimeError("model forward returned no extra_info dict; cannot score holdout KL")
                kl_sum += float(extra["reduced_qad_kl_sum"].sum().item())
                token_sum += float(extra["reduced_qad_valid_count"].sum().item())
    if token_sum == 0.0:
        return {"kl": float("nan"), "tokens": 0.0, "kl_sum": 0.0}
    return {"kl": kl_sum / token_sum, "tokens": token_sum, "kl_sum": kl_sum}


def clone_student_cpu(model_cfg: Qwen3DenseConfig, trainer: QADTrainer, quant_cfg: QADQuantConfig) -> Any:
    """Full-precision CPU clone of the trained student, carrying the trainer's frozen scale
    caches verbatim (gate C premise: deployment scale == training-end scale)."""
    # Build directly in bf16 (the loaded state is bf16): avoids an fp32-sized transient that
    # matters on this 60GB-RAM host. Cast PARAMETERS only — buffers such as the rotary
    # ``inv_freq`` stay fp32, matching the training-time forward (a whole-module bf16 cast
    # degrades rope angles and breaks gate C bit-exactness; measured).
    with torch.device("cpu"):
        plain = model_cfg.build()
    for _name, param in plain.named_parameters():
        param.data = param.data.to(torch.bfloat16)
    state = {k: _full(v) for k, v in trainer._engine.model.state_dict().items()}
    missing, unexpected = plain.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"clone load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    for _name, param in plain.named_parameters():
        param.data = param.data.to(torch.bfloat16)
    apply_quantization(plain, quant_cfg)
    if quant_cfg.kv_cache is not None:
        for _name, attention in _attention_modules(plain):
            apply_kv_quantization(attention, quant_cfg.kv_cache, quant_cfg.kv_amax_ema_decay)
    src_modules: dict[str, FakeQuantLinear] = {}
    other_fq_names: list[str] = []
    for name, module in trainer._engine.model.named_modules():
        if isinstance(module, FakeQuantLinear):
            src_modules[name] = module
        elif "q_proj" in name:
            other_fq_names.append(f"{name}({type(module).__name__})")
    if not src_modules:
        probe = [f"{n}:{type(m).__name__}" for n, m in list(trainer._engine.model.named_modules())[:15]]
        raise RuntimeError(
            "clone: no FakeQuantLinear found in trainer model; module tree probe: "
            f"{probe}; q_proj-named entries: {other_fq_names[:5]}"
        )
    plain_fq = [(name, module) for name, module in plain.named_modules() if isinstance(module, FakeQuantLinear)]
    src_fq = list(src_modules.items())
    if {name for name, _ in plain_fq} != set(src_modules):
        # Identical trees built from one config; pair in named_modules order as a fallback.
        if len(plain_fq) != len(src_fq):
            raise RuntimeError(
                f"clone: FakeQuantLinear count mismatch trainer={len(src_fq)} plain={len(plain_fq)}; "
                f"trainer probe={other_fq_names[:5]}"
            )
        src_modules = {p_name: s_mod for (p_name, _s), (_s_name, s_mod) in zip(plain_fq, src_fq)}
    for name, module in plain_fq:
        src = src_modules[name]
        if not isinstance(src, FakeQuantLinear):
            raise RuntimeError(f"clone: source FakeQuantLinear '{name}' not found in trainer model")
        module._scales_frozen = bool(src._scales_frozen)
        if src._w_scale_cache is not None:
            # fp32 on purpose: the trainer computes weight scales in fp32 (§4.3) and export
            # stores fp32 values; rounding them to the (bf16) param dtype desynchronizes the
            # clone's fake-quant from both the trainer and the replayed export.
            module._w_scale_cache = _full(src._w_scale_cache).float()
    src_handle_by_name: dict[str, KVQuantHandle] = {}
    for t_name, t_module in trainer._engine.model.named_modules():
        t_handle = getattr(t_module, "_qad_kv_handle", None)
        if isinstance(t_handle, KVQuantHandle):
            src_handle_by_name[t_name] = t_handle
    plain_kv = [
        (p_name, p_module)
        for p_name, p_module in plain.named_modules()
        if isinstance(p_module, nn.Module) and getattr(p_module, "_qad_kv_handle", None) is not None
    ]
    plain_kv_names = {p_name for p_name, _ in plain_kv}
    if plain_kv_names != set(src_handle_by_name):
        if len(plain_kv) != len(src_handle_by_name):
            raise RuntimeError(
                f"clone: KV handle count mismatch trainer={len(src_handle_by_name)} plain={len(plain_kv)}"
            )
        src_handle_by_name = {p_name: list(src_handle_by_name.values())[i] for i, (p_name, _m) in enumerate(plain_kv)}
    for p_name, p_module in plain_kv:
        handle = getattr(p_module, "_qad_kv_handle")
        src_handle = src_handle_by_name[p_name]
        src_q = src_handle.quantizer
        if src_q.k_amax_ema is not None:
            handle.quantizer.k_amax_ema = src_q.k_amax_ema.detach().float().clone()
        if src_q.v_amax_ema is not None:
            handle.quantizer.v_amax_ema = src_q.v_amax_ema.detach().float().clone()
        handle.quantizer.set_frozen(src_q.frozen)
    return plain


def replay_model_gpu(model_cfg: Qwen3DenseConfig, export_dir: Path, loss_cfg: QADLossConfig, ema_source: Any) -> Any:
    """Fresh model + exported master weights + load_qad_export metadata replay (gate C)."""
    from safetensors.torch import load_file

    # The fresh model must carry the QAD loss config, or build_loss_ctx_batch creates a plain
    # CE loss context whose extra_info lacks the qad metrics the KL scoring reads.
    replay_cfg = model_cfg.model_copy(deep=True)
    replay_cfg.lm_loss_cfg = loss_cfg
    with torch.device(DEVICE):
        model = replay_cfg.build()
    weights = load_file(str(export_dir / "model.safetensors"), device=str(DEVICE))
    missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"replay load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    # ``load_qad_export`` applies KV simulation with a full-model name walk whose pattern
    # (path contains "attention") also matches e.g. ``post_attention_layernorm`` (an RMSNorm
    # with no k_proj). Pre-mark every non-Linear norm with an unpatched KVQuantHandle so the
    # walk's idempotency check skips them; the real attention modules are then patched exactly
    # as in the training run. Handles are removed afterwards to keep the replay tree clean.
    from xtuner.v1.quant.kv_cache import _HANDLE_ATTR, KVCacheQuantizer, KVQuantHandle

    guarded: list[tuple[nn.Module, Any]] = []
    for name, module in model.named_modules():
        lowered = name.lower()
        cls_name = type(module).__name__
        name_hit = "self_attn" in lowered or "attention" in lowered
        class_hit = "Attention" in cls_name or "MLA" in cls_name
        if not (name_hit or class_hit) or isinstance(module, nn.Linear):
            continue
        has_proj = isinstance(getattr(module, "k_proj", None), nn.Linear) or isinstance(
            getattr(module, "v_proj", None), nn.Linear
        )
        if has_proj:
            continue  # real attention module: leave it for load_qad_export to patch
        handle = KVQuantHandle(module, KVCacheQuantizer())
        setattr(module, _HANDLE_ATTR, handle)
        guarded.append((module, handle))
    load_qad_export(model, export_dir / "qad_quant_metadata.json")
    for module, _handle in guarded:
        if getattr(module, _HANDLE_ATTR, None) is _handle:
            delattr(module, _HANDLE_ATTR)
    # load_qad_export injects scales as CPU tensors (JSON round-trip); the training model
    # computes them on-device. Move them to the replay device — values unchanged, so the
    # replay stays bit-identical to the exporting run.
    # Known T8 replay-loader artifact (reported): _inject_kv_scales reconstructs
    # ``k_amax_ema = scale * 127``, and ``clamp(ema / 127)`` recovers the scale 1 ULP off,
    # which flips quantization roundings at value boundaries. Re-pin the exporting run's
    # exact EMA tensors (the metadata scale VALUES themselves are exact fp32).
    src_ema = {
        n: q
        for n, q in ((n, getattr(m, "_qad_kv_quantizer", None)) for n, m in ema_source.named_modules())
        if q is not None
    }
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantLinear) and module._w_scale_cache is not None:
            module._w_scale_cache = module._w_scale_cache.to(DEVICE)
        quantizer = getattr(module, "_qad_kv_quantizer", None)
        if quantizer is None:
            continue
        src_q = src_ema.get(name)
        if src_q is not None:
            if src_q.k_amax_ema is not None:
                quantizer.k_amax_ema = src_q.k_amax_ema.detach().float().to(DEVICE)
            if src_q.v_amax_ema is not None:
                quantizer.v_amax_ema = src_q.v_amax_ema.detach().float().to(DEVICE)
        else:
            if quantizer.k_amax_ema is not None:
                quantizer.k_amax_ema = quantizer.k_amax_ema.to(DEVICE)
            if quantizer.v_amax_ema is not None:
                quantizer.v_amax_ema = quantizer.v_amax_ema.to(DEVICE)
    return model


def run_experiment(args: argparse.Namespace, schema: str, kv_dtype: str) -> dict[str, Any]:
    exp_name = f"{schema}-kv{kv_dtype}"
    exp_dir = Path(args.work_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    kv_cfg = KVQConfig(dtype="int8") if kv_dtype == "int8" else None
    model_cfg = _load_model_cfg(args.model_dir)
    model_cfg.compile_cfg = False

    quant_cfg = QADQuantConfig.for_schema(
        schema,
        kv_cache=kv_cfg,
        kv_amax_ema_decay=args.kv_ema_decay,
        scale_freeze_last_ratio=args.freeze_ratio,
    )
    teacher_cfg = QADTeacherAdapterConfig(
        teacher=TrainTeacherConfig(
            name="teacher",
            model_path=args.model_dir,
            model_cfg=_load_model_cfg(args.model_dir),
            fsdp_cfg=FSDPConfig(tp_size=1, recompute_ratio=0, torch_compile=False, requires_grad=False),
        ),
        teacher_offload=bool(args.teacher_offload),
        chunk_size=args.teacher_chunk,
    )
    dataset_cfg = [
        {
            "dataset": DatasetConfig(name="qad_sim_train", anno_path=args.train_jsonl, sample_ratio=1.0),
            "tokenize_fn": OpenaiTokenizeFunctionConfig(chat_template="intern-s1", max_length=args.pack_max_length),
        }
    ]
    kl_curve: list[dict[str, float]] = []

    def holdout_hook(
        train_step_info: dict, step: int, epoch: int | None, total_step: int, total_epoch: int | None
    ) -> None:
        del train_step_info, total_step, total_epoch
        if (step + 1) % args.eval_interval != 0:
            return
        t0 = time.time()
        kl = trainer.evaluate_holdout(holdout_batches)  # noqa: F821 -- late-bound closure
        kl_curve.append({"step": step + 1, "holdout_kl": kl, "eval_time_s": round(time.time() - t0, 2)})
        trainer.logger.info(f"[{exp_name}] holdout KL @ step {step + 1}: {kl:.6f}")  # noqa: F821 -- late-bound closure

    trainer_cfg = TrainerConfig(
        model_cfg=model_cfg,
        optim_cfg=AdamWConfig(lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), foreach=False, swap_optimizer=True),
        fsdp_cfg=FSDPConfig(tp_size=1, recompute_ratio=1.0, torch_compile=False),
        dataloader_cfg=DataloaderConfig(
            dataset_config_list=dataset_cfg,
            pack_level="soft",
            pack_max_length=args.pack_max_length,
            pack_chunk_size=1,
            pack_workers=1,
            global_pack=False,
            num_workers=0,
        ),
        lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0.0, lr_min=args.lr),
        global_batch_size=args.global_batch_size,
        total_step=args.total_steps,
        work_dir=exp_dir / "trainer",
        tokenizer_path=args.model_dir,
        checkpoint_interval=None,
        hf_interval=None,
        snapshot_interval=None,
        hf_max_keep=None,
        check_health_interval=None,
        exp_tracker="jsonl",
        profile_time=False,
        profile_memory=False,
        hooks_config=HooksConfig(after_train_step=holdout_hook),
    )
    loss_cfg = QADLossConfig(
        mode="qad_topk_forward",
        top_k=args.top_k,
        temperature=1.0,
        log_prob_min_clamp=-10.0,
        loss_max_clamp=10.0,
    )
    qad_cfg = QADTrainerConfig(
        trainer=trainer_cfg,
        quant=quant_cfg,
        teacher=teacher_cfg,
        loss=loss_cfg,
        model_path=args.model_dir,
        from_hub="modelscope",
        trainable="full",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    holdout_batches = build_holdout_batches(
        Path(args.holdout_jsonl), tokenizer, args.pack_max_length, pad_id, args.holdout_rows
    )
    n_holdout_tokens = sum(int(b[0]["seq_ctx"].input_ids.shape[-1]) for b in holdout_batches)

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    trainer = QADTrainer.from_config(qad_cfg)

    # --- Gate A anchor: teacher vs initial-weight BF16 student must be exactly 0. ----------
    kl_a = _gate_a_anchor(trainer, holdout_batches)

    # --- PTQ baseline (fake-quant forward with amax scales, no training). ------------------
    t0 = time.time()
    kl_ptq = trainer.evaluate_holdout(holdout_batches)
    ptq_time = time.time() - t0
    kl_curve.append({"step": 0, "holdout_kl": kl_ptq, "eval_time_s": round(ptq_time, 2)})
    trainer.logger.info(f"[{exp_name}] PTQ baseline holdout KL: {kl_ptq:.6f}")

    # --- QAD micro-run. ---------------------------------------------------------------------
    t0 = time.time()
    trainer.fit()
    train_time = time.time() - t0

    # --- KL(B_end): end of training, scale-freeze phase engaged. ----------------------------
    kl_b_end = trainer.evaluate_holdout(holdout_batches)
    kl_curve.append({"step": args.total_steps, "holdout_kl": kl_b_end, "eval_time_s": None})

    # --- Gate C: export -> replay. ----------------------------------------------------------
    trainer._engine.optimizer.zero_grad(set_to_none=True)
    export_dir = exp_dir / "export"
    plain = clone_student_cpu(model_cfg, trainer, quant_cfg)
    report = export_qad(plain, quant_cfg, export_dir)
    # Debug: score the clone itself on the first holdout row to localize gate C residuals
    # (trainer bf16 student vs clone vs replayed export).
    plain_gpu = plain.to(DEVICE)
    with torch.no_grad():
        ditem = dict(holdout_batches[0][0])
        dtargets = trainer._teacher_targets(ditem["seq_ctx"], loss_cfg)
        ditem["teacher_logprobs"] = dtargets["teacher_logprobs"]
        if "target_token_ids" in dtargets:
            ditem["target_token_ids"] = dtargets["target_token_ids"]
        ditem["loss_mask"] = _loss_mask(ditem, loss_cfg)
        dctx = plain_gpu.build_loss_ctx_batch([ditem], sp_mesh=trainer.sp_mesh)[0]
        dout = plain_gpu(seq_ctx=ditem["seq_ctx"].to(DEVICE), loss_ctx={"lm": dctx["lm"]})
        logits_clone = _full(dout.logits)

    def _layer_stats(model: Any) -> dict[str, tuple[float, float]]:
        stats: dict[str, tuple[float, float]] = {}

        def _mk(key: str):
            def hook(_mod: Any, _inp: Any, out: Any) -> None:
                h = out["hidden_states"] if isinstance(out, dict) else out[0] if isinstance(out, tuple) else out
                if isinstance(h, DTensor):
                    h = h.full_tensor()
                stats[key] = (h.detach().abs().max().float().item(), h.detach().float().abs().mean().item())

            return hook

        handles = [model.embed_tokens.register_forward_hook(_mk("embed"))]
        for idx_str, layer in model.layers.items():
            handles.append(layer.register_forward_hook(_mk(f"layer{idx_str}")))
        handles.append(model.norm.register_forward_hook(_mk("final_norm")))
        return stats, handles

    t_stats, t_handles = _layer_stats(trainer._engine.model)
    c_stats, c_handles = _layer_stats(plain_gpu)
    # Layer0 submodule localization: attention vs MLP vs norms.
    for tag, model in (("train", trainer._engine.model), ("clone", plain_gpu)):
        l0 = model.layers["0"]

        def _sub_hook(key: str):
            def hook(_mod: Any, _inp: Any, out: Any) -> None:
                if isinstance(out, dict):
                    h = out.get("hidden_states")
                    if h is None:
                        return
                elif isinstance(out, tuple) and out:
                    h = out[0]
                else:
                    h = out
                if not isinstance(h, torch.Tensor):
                    return
                if isinstance(h, DTensor):
                    h = h.full_tensor()
                t_stats[f"l0.{key}"] = (h.detach().abs().max().float().item(), h.detach().float().abs().mean().item())

            return hook

        t_handles.append(l0.input_layernorm.register_forward_hook(_sub_hook(f"input_layernorm/{tag}")))
        t_handles.append(l0.self_attn.register_forward_hook(_sub_hook(f"self_attn/{tag}")))
        t_handles.append(l0.post_attention_layernorm.register_forward_hook(_sub_hook(f"post_attn_ln/{tag}")))
        t_handles.append(l0.mlp.register_forward_hook(_sub_hook(f"mlp/{tag}")))
    # Direct weight-level comparison (catches dtype/rounding mismatches before any forward).
    t_sd = {k: _full(v) for k, v in trainer._engine.model.state_dict().items()}
    for key in ("embed_tokens.weight", "lm_head.weight", "layers.0.self_attn.q_proj.weight", "norm.weight"):
        t_w = t_sd.get(key)
        c_named = dict(plain_gpu.named_parameters())
        c_w = c_named.get(key)
        if t_w is None or c_w is None:
            trainer.logger.info(
                f"[{exp_name}] weight-diff {key}: MISSING (train={t_w is not None} clone={c_w is not None})"
            )
            continue
        t32, c32 = t_w.float(), c_w.float()
        trainer.logger.info(
            f"[{exp_name}] weight-diff {key}: dtypes train={t_w.dtype} clone={c_w.dtype} "
            f"maxdiff={(t32 - c32).abs().max().item():.3e} train_absmax={t32.abs().max().item():.4f}"
        )
    del t_sd
    # Rope check: the trainer builds on the meta device and Dense.from_hf's rebuild of the
    # non-persistent rotary inv_freq is commented out; a fresh build computes it from theta.
    t_rot = getattr(trainer._engine.model, "rotary_emb", None)
    c_rot = getattr(plain_gpu, "rotary_emb", None)
    for side, rot in (("train", t_rot), ("clone", c_rot)):
        inv = getattr(rot, "inv_freq", None) if rot is not None else None
        if inv is None:
            trainer.logger.info(f"[{exp_name}] rotary[{side}]: no inv_freq attr ({type(rot).__name__})")
        else:
            trainer.logger.info(
                f"[{exp_name}] rotary[{side}]: shape={tuple(inv.shape)} first3={inv.flatten()[:3].tolist()} absmax={inv.abs().max().item():.4f}"
            )
    with torch.no_grad():
        for model, _handles in ((trainer._engine.model, t_handles), (plain_gpu, c_handles)):
            model(seq_ctx=ditem["seq_ctx"].to(DEVICE), loss_ctx=None)
    for key in t_stats:
        t_max, t_mean = t_stats[key]
        c_max, c_mean = c_stats.get(key, (float("nan"), float("nan")))
        trainer.logger.info(
            f"[{exp_name}] layerwise {key}: train(max={t_max:.4f} mean={t_mean:.4f}) clone(max={c_max:.4f} mean={c_mean:.4f})"
        )
    for h in t_handles + c_handles:
        h.remove()
    replay = replay_model_gpu(model_cfg, export_dir, loss_cfg, ema_source=plain_gpu)
    rc = holdout_kl_for_model(trainer, replay, holdout_batches, sp_mesh=trainer.sp_mesh)
    kl_c = rc["kl"]
    kl_c_sum = rc["kl_sum"]

    # Bit-level logits comparison on the first holdout row (training student vs replay).
    loss_cfg = trainer._qad_loss_cfg()
    with torch.no_grad():
        item = dict(holdout_batches[0][0])
        targets = trainer._teacher_targets(item["seq_ctx"], loss_cfg)
        item["teacher_logprobs"] = targets["teacher_logprobs"]
        item["target_token_ids"] = targets["target_token_ids"]
        item["loss_mask"] = _loss_mask(item, loss_cfg)
        ctx_train = trainer._engine.model.build_loss_ctx_batch([item], sp_mesh=trainer.sp_mesh)[0]
        out_train = trainer._engine.model(seq_ctx=item["seq_ctx"].to(DEVICE), loss_ctx={"lm": ctx_train["lm"]})
        logits_train = _full(out_train.logits)
        ctx_replay = replay.build_loss_ctx_batch([item], sp_mesh=None)[0]
        out_replay = replay(seq_ctx=item["seq_ctx"].to(DEVICE), loss_ctx={"lm": ctx_replay["lm"]})
        logits_replay = _full(out_replay.logits)
    # torch.equal is dtype-strict: the FSDP path returns fp32 logits while the plain clone /
    # replay keep bf16; compare values in fp32 (max_abs_diff independently guards numerics).
    logits_equal = bool(torch.equal(logits_train.float(), logits_replay.float()))
    logits_max_abs_diff = float((logits_train.float() - logits_replay.float()).abs().max().item())
    clone_vs_train = float((logits_clone.float() - logits_train.float()).abs().max().item())
    clone_vs_replay = float((logits_clone.float() - logits_replay.float()).abs().max().item())
    clone_equal_train = bool(torch.equal(logits_clone.float(), logits_train.float()))
    clone_equal_replay = bool(torch.equal(logits_clone.float(), logits_replay.float()))
    # Full-state diff: weights, scale caches, KV EMA buffers, rope.
    with torch.no_grad():
        c_sd = {k: _full(v) for k, v in plain_gpu.state_dict().items()}
        r_sd = {k: _full(v) for k, v in replay.state_dict().items()}
        weight_diffs = []
        for key in sorted(set(c_sd) | set(r_sd)):
            a, b = c_sd.get(key), r_sd.get(key)
            if a is None or b is None:
                weight_diffs.append(f"{key}:MISSING")
            elif a.shape != b.shape or a.dtype != b.dtype:
                weight_diffs.append(f"{key}:META({a.dtype}/{b.dtype},{tuple(a.shape)}/{tuple(b.shape)})")
            elif not torch.equal(a, b):
                weight_diffs.append(f"{key}:VAL({(a.float() - b.float()).abs().max().item():.3e})")
        c_cache = {
            n: m._w_scale_cache.detach().float().cpu()
            for n, m in plain_gpu.named_modules()
            if isinstance(m, FakeQuantLinear) and m._w_scale_cache is not None
        }
        r_cache = {
            n: m._w_scale_cache.detach().float().cpu()
            for n, m in replay.named_modules()
            if isinstance(m, FakeQuantLinear) and m._w_scale_cache is not None
        }
        cache_diffs = [
            n
            for n in sorted(set(c_cache) | set(r_cache))
            if n not in c_cache or n not in r_cache or not torch.equal(c_cache[n], r_cache[n])
        ]
        c_ema = {
            n: (q.k_amax_ema, q.v_amax_ema)
            for n, q in ((n, getattr(m, "_qad_kv_quantizer", None)) for n, m in plain_gpu.named_modules())
            if q is not None
        }
        r_ema = {
            n: (q.k_amax_ema, q.v_amax_ema)
            for n, q in ((n, getattr(m, "_qad_kv_quantizer", None)) for n, m in replay.named_modules())
            if q is not None
        }
        ema_diffs = []
        for n in sorted(set(c_ema) | set(r_ema)):
            a, b = c_ema.get(n), r_ema.get(n)
            if a is None or b is None:
                ema_diffs.append(f"{n}:MISSING")
                continue
            for tag, ta, tb in (("k", a[0], b[0]), ("v", a[1], b[1])):
                if (ta is None) != (tb is None):
                    ema_diffs.append(f"{n}.{tag}:NONE_MISMATCH")
                elif ta is not None:
                    d = (ta.float().cpu() - tb.float().cpu()).abs().max().item()
                    if d > 0:
                        ema_diffs.append(
                            f"{n}.{tag}:maxdiff={d:.3e} train_amax={ta.float().max().item():.3e} shape={tuple(ta.shape)}/{tuple(tb.shape)}"
                        )
        trainer.logger.info(
            f"[{exp_name}] state-diff: weights={len(weight_diffs)} {weight_diffs[:6]} | caches={len(cache_diffs)} {cache_diffs[:4]} | ema={len(ema_diffs)} {ema_diffs[:6]}"
        )
        del c_sd, r_sd
    del plain_gpu
    del plain
    trainer.logger.info(
        f"[{exp_name}] debug: clone-vs-train equal={clone_equal_train} maxdiff={clone_vs_train:.4e} | "
        f"clone-vs-replay equal={clone_equal_replay} maxdiff={clone_vs_replay:.4e}"
    )

    peak_mem_gib = torch.cuda.max_memory_allocated() / 2**30

    gate_b = bool(kl_b_end < kl_ptq)
    kl_c_abs_diff = abs(kl_c - kl_b_end)
    gate_c = bool(
        logits_equal or (kl_c_abs_diff <= args.gate_c_tol and math.isfinite(kl_c) and math.isfinite(kl_b_end))
    )
    result = {
        "experiment": exp_name,
        "schema": schema,
        "kv_cache": kv_dtype,
        "trainable": "full",
        "model": {
            "path": args.model_dir,
            "num_layers": model_cfg.num_hidden_layers,
            "hidden_size": model_cfg.hidden_size,
            "vocab_size": model_cfg.vocab_size,
            "num_parameters": sum(p.numel() for p in trainer._engine.model.parameters()),
        },
        "quant": quant_cfg.model_dump(),
        "loss": {"mode": "qad_topk_forward", "top_k": args.top_k, "temperature": 1.0},
        "steps": args.total_steps,
        "lr": args.lr,
        "global_batch_size": args.global_batch_size,
        "pack_max_length": args.pack_max_length,
        "holdout": {
            "rows": len(holdout_batches),
            "tokens_per_eval": n_holdout_tokens,
            "jsonl": str(args.holdout_jsonl),
        },
        "ptq_init_kl": kl_a,  # quantized student at init (fake-quant already applied) ~= PTQ baseline;
        # NOT the BF16 anchor KL(A)=0, which gate_a_anchor.py verifies independently. Historically
        # mislabeled "gate_a_kl" (see report.md §6.8).
        "kl_ptq": kl_ptq,
        "kl_b_end": kl_b_end,
        "kl_c": kl_c,
        "kl_c_sum": kl_c_sum,
        "kl_c_abs_diff": kl_c_abs_diff,
        "kl_curve": kl_curve,
        "gate_b": gate_b,
        "gate_c": gate_c,
        "gate_c_tol": args.gate_c_tol,
        "env": {
            "XTUNER_HF_IMPL": os.environ.get("XTUNER_HF_IMPL", ""),
            "XTUNER_USE_FA3": os.environ.get("XTUNER_USE_FA3", ""),
            "flash_attn_3_installed": importlib.util.find_spec("flash_attn_3") is not None
            or importlib.util.find_spec("flash_attn_interface") is not None,
        },
        "logits_equal_frozen": logits_equal,
        "logits_max_abs_diff": logits_max_abs_diff,
        "export": {
            k: report[k]
            for k in ("weights_path", "metadata_path", "num_modules", "num_kv_handles", "model_vocab_size")
        },
        "timing_s": {
            "train": round(train_time, 1),
            "ptq_eval": round(ptq_time, 2),
            "total": round(time.time() - t_start, 1),
        },
        "peak_gpu_mem_gib": round(peak_mem_gib, 2),
    }
    (exp_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    trainer.logger.info(
        f"[{exp_name}] DONE: PTQ={kl_ptq:.6f} B_end={kl_b_end:.6f} C={kl_c:.6f} "
        f"gate_b={gate_b} gate_c={gate_c} logits_equal={logits_equal} diff={logits_max_abs_diff:.3e}"
    )
    del trainer, replay
    torch.cuda.empty_cache()
    return result


def _gate_a_anchor(trainer: QADTrainer, holdout_batches: list[list[dict[str, Any]]]) -> float:
    """KL(A): bf16 initial-weight student vs teacher on the holdout (must be 0)."""
    loss_cfg = trainer._qad_loss_cfg()
    kl_sum = 0.0
    token_sum = 0.0
    with torch.no_grad():
        for data_batch in holdout_batches:
            for item in data_batch:
                targets = trainer._teacher_targets(item["seq_ctx"], loss_cfg)
                item["teacher_logprobs"] = targets["teacher_logprobs"]
                if "target_token_ids" in targets:
                    item["target_token_ids"] = targets["target_token_ids"]
                item["loss_mask"] = _loss_mask(item, loss_cfg)
            ctx_list = trainer._engine.model.build_loss_ctx_batch(data_batch, sp_mesh=trainer.sp_mesh)
            for item, ctx in zip(data_batch, ctx_list):
                output = trainer._engine.forward_only(item["seq_ctx"].to(DEVICE), ctx["lm"])
                extra = getattr(output, "extra_info", None)
                kl_sum += float(extra["reduced_qad_kl_sum"].sum().item())
                token_sum += float(extra["reduced_qad_valid_count"].sum().item())
    return kl_sum / token_sum if token_sum else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="w8a8:int8,w4a8:int8,w4a8:none")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--total-steps", type=int, default=240)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--pack-max-length", type=int, default=2048)
    parser.add_argument("--eval-interval", type=int, default=45)
    parser.add_argument("--holdout-rows", type=int, default=8)
    parser.add_argument("--freeze-ratio", type=float, default=0.1)
    parser.add_argument("--kv-ema-decay", type=float, default=0.999)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--teacher-chunk", type=int, default=512)
    parser.add_argument(
        "--teacher-offload", type=int, default=1, help="1: teacher CPU offload (memory-safe); 0: resident on GPU"
    )
    parser.add_argument("--gate-c-tol", type=float, default=1e-6)
    args = parser.parse_args()

    results = []
    for spec in args.experiments.split(","):
        schema, kv_dtype = spec.split(":")
        results.append(run_experiment(args, schema, kv_dtype))
    out = Path(args.work_dir) / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged: list[dict] = []
    if out.exists():
        try:
            merged = [
                r for r in json.loads(out.read_text(encoding="utf-8")) if isinstance(r, dict) and "experiment" in r
            ]
        except json.JSONDecodeError:
            merged = []
    by_name = {r.get("experiment"): r for r in merged}
    for r in results:
        by_name[r["experiment"]] = r
    out.write_text(json.dumps(list(by_name.values()), indent=2), encoding="utf-8")
    print(f"[run_experiments] wrote {out}")


if __name__ == "__main__":
    main()
