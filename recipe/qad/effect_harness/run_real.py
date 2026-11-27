# Effect-lane: REAL-checkpoint QAD run — Qwen3-0.6B (ModelScope), w8a8+kv8, 500 steps.
#
# Pipeline (assignment): PTQ baseline holdout KL (gate B denominator) -> QAD run (LR 1e-5
# constant, teacher resident, train KL every 10 steps / holdout KL every 50) -> ModelOpt-lane
# assertions -> export/replay gate C (bit-exact) -> greedy generation on a fixed 30-prompt
# probe set across BF16 / PTQ / QAD.
#
# Runtime adaptations for the REAL Qwen3 architecture (no xtuner/v1 source edits):
# - qk_norm=True (real checkpoint carries q_norm/k_norm RMSNorms inside self_attn). The stock
#   KV-quant attachment matches modules by name ("self_attn") and raises on modules without a
#   K/V entry point, so the attachment is wrapped with a norm-safe guard that pre-marks
#   q_norm/k_norm (no k_proj/v_proj) and unmarks them afterwards — mirroring the guard that
#   work_dir/qad_sim/run_experiment.py already applies to post_attention_layernorm in replay.
# - tie_word_embeddings=True is kept as-is (Dense.fully_shard re-ties after FSDP wrapping).
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor

import xtuner.v1.quant.export as _export_mod
import xtuner.v1.quant.kv_cache as _kv_cache_mod
import xtuner.v1.train.qad_trainer as _qad_trainer_mod
from xtuner.v1.quant.kv_cache import _HANDLE_ATTR, KVCacheQuantizer, KVQuantHandle, _is_kv_quantizable


# --- norm-safe KV-quant attachment (real Qwen3 carries q_norm/k_norm) -----------------------


def _has_kv_entry(module: nn.Module) -> bool:
    return any(isinstance(getattr(module, n, None), nn.Linear) for n in ("k_proj", "v_proj", "kv_b_proj"))


_orig_apply_kv = _kv_cache_mod.apply_kv_quantization


def _apply_kv_normsafe(model: nn.Module, kv_cfg: Any, kv_amax_ema_decay: float = 0.999) -> list[KVQuantHandle]:
    """``apply_kv_quantization`` with real-Qwen3 qk_norm support.

    Name-matched modules that expose no K/V entry point (``self_attn.q_norm``/``k_norm``
    RMSNorms of the real checkpoint) are temporarily pre-guarded so the stock walk's
    idempotency check skips them instead of raising on the missing ``k_proj``.
    """
    if kv_cfg is None:
        return []
    guarded: list[tuple[nn.Module, KVQuantHandle]] = []
    for _name, module in list(model.named_modules()):
        if _is_kv_quantizable(_name, module) and not _has_kv_entry(module):
            handle = KVQuantHandle(module, KVCacheQuantizer(ema_decay=kv_amax_ema_decay))
            setattr(module, _HANDLE_ATTR, handle)
            guarded.append((module, handle))
    try:
        return _orig_apply_kv(model, kv_cfg, kv_amax_ema_decay)
    finally:
        for module, handle in guarded:
            if getattr(module, _HANDLE_ATTR, None) is handle:
                delattr(module, _HANDLE_ATTR)


_kv_cache_mod.apply_kv_quantization = _apply_kv_normsafe
_export_mod.apply_kv_quantization = _apply_kv_normsafe

_orig_attn_modules = _qad_trainer_mod._attention_modules


def _attention_modules_normsafe(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(name, module) for name, module in _orig_attn_modules(model) if _has_kv_entry(module)]


_qad_trainer_mod._attention_modules = _attention_modules_normsafe

# --- shared sim-harness helpers (import AFTER the patches so stale bindings can't persist) --
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # work_dir/qad_sim
import run_experiment as R  # noqa: E402

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig  # noqa: E402
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig  # noqa: E402
from xtuner.v1.datasets.sft_tokenize_fn import OpenaiTokenizeFunctionConfig  # noqa: E402
from xtuner.v1.model import get_model_config_from_hf  # noqa: E402
from xtuner.v1.quant.config import KVQConfig, QADQuantConfig  # noqa: E402
from xtuner.v1.quant.export import export_qad  # noqa: E402
from xtuner.v1.quant.loss import QADLossConfig  # noqa: E402
from xtuner.v1.quant.teacher import QADTeacherAdapterConfig  # noqa: E402
from xtuner.v1.rl.distillation.config import TrainTeacherConfig  # noqa: E402
from xtuner.v1.train.qad_trainer import QADTrainer, QADTrainerConfig, _loss_mask  # noqa: E402
from xtuner.v1.train.trainer import HooksConfig, TrainerConfig  # noqa: E402
from xtuner.v1.utils.device import get_device  # noqa: E402


DEVICE = get_device()
PROBE_PROMPTS = None  # filled by build_generation_probe


def _full(t: torch.Tensor | Any) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t.detach().clone()


def load_model_cfg(model_dir: str, tie_word_embeddings: bool | None = None) -> Any:
    """Real Qwen3 config: keep qk_norm=True (real architecture); compile off."""
    model_cfg = get_model_config_from_hf(Path(model_dir))
    model_cfg.compile_cfg = False
    if tie_word_embeddings is not None:
        model_cfg.tie_word_embeddings = tie_word_embeddings
    return model_cfg


def gate_a_anchor_independent(model_dir: str, model_cfg: Any, probe_row_ids: list[int]) -> dict:
    """Independent gate A: two fresh BF16 instances of the same checkpoint must agree bitwise.

    The teacher IS a BF16 copy of the initial student, so KL(A)=0 by construction; this
    verifies the premise numerically (the in-trainer anchor measures the fake-quant student,
    historically mislabeled — see report.md §6.8).
    """
    from xtuner.v1.data_proto import SequenceContext

    ids = torch.tensor([probe_row_ids], dtype=torch.long)
    outs = []
    for _ in range(2):
        with torch.device(DEVICE):
            model = model_cfg.model_copy(deep=True).build()
        _loaded, _unloaded, missing = model.from_hf(model_dir, strict=False)
        assert not missing, f"gate A anchor: missing keys {sorted(missing)[:5]}"
        seq_ctx = SequenceContext(
            input_ids=ids,
            cu_seq_lens_q=torch.tensor([0, ids.shape[1]], dtype=torch.int),
            cu_seq_lens_k=torch.tensor([0, ids.shape[1]], dtype=torch.int),
            max_length_q=ids.shape[1],
            max_length_k=ids.shape[1],
            device=str(DEVICE),
        ).to(DEVICE)
        with torch.no_grad():
            out = model(seq_ctx=seq_ctx, loss_ctx=None)
        outs.append(_full(out.logits).float().cpu())
        del model
        torch.cuda.empty_cache()
    return {
        "bitwise_equal": bool(torch.equal(outs[0], outs[1])),
        "max_abs_diff": float((outs[0] - outs[1]).abs().max().item()),
        "tokens": int(ids.shape[1]),
    }


def run_qad(
    model_dir: str,
    train_jsonl: str,
    holdout_jsonl: str,
    exp_dir: Path,
    schema: str = "w8a8",
    kv_dtype: str = "int8",
    total_steps: int = 500,
    lr: float = 1e-5,
    train_kl_interval: int = 10,
    holdout_interval: int = 50,
    pack_max_length: int = 1024,
    holdout_rows: int = 64,
    freeze_ratio: float = 0.1,
    kv_ema_decay: float = 0.999,
    top_k: int = 64,
    teacher_offload: bool = False,
    generation: bool = True,
    max_gen_tokens: int = 96,
    chat_template: str = "qwen3",
    global_batch_size: int = 1,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    """One real-model QAD experiment: PTQ baseline -> run -> gates -> export -> replay."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    kv_cfg = KVQConfig(dtype="int8") if kv_dtype == "int8" else None
    model_cfg = load_model_cfg(model_dir)

    quant_cfg = QADQuantConfig.for_schema(
        schema, kv_cache=kv_cfg, kv_amax_ema_decay=kv_ema_decay, scale_freeze_last_ratio=freeze_ratio
    )
    teacher_cfg = QADTeacherAdapterConfig(
        teacher=TrainTeacherConfig(
            name="teacher",
            model_path=model_dir,
            model_cfg=load_model_cfg(model_dir),
            fsdp_cfg=FSDPConfig(tp_size=1, recompute_ratio=1.0, torch_compile=False, requires_grad=False),
        ),
        teacher_offload=teacher_offload,
        chunk_size=512,
    )
    dataset_cfg = [
        {
            "dataset": DatasetConfig(name="qad_effect_train", anno_path=train_jsonl, sample_ratio=1.0),
            "tokenize_fn": OpenaiTokenizeFunctionConfig(chat_template=chat_template, max_length=pack_max_length),
        }
    ]

    train_kl_curve: list[dict[str, float]] = []
    kl10_curve: list[dict[str, float]] = []
    kl_window: list[float] = []
    cum_kl_curve: list[dict[str, float]] = []
    holdout_kl_curve: list[dict[str, float]] = []
    loss_curve: list[dict[str, float]] = []

    def after_step(
        train_step_info: dict, step: int, epoch: int | None, total_step: int, total_epoch: int | None
    ) -> None:
        del epoch, total_epoch
        extra = train_step_info.get("extra_info") if isinstance(train_step_info, dict) else None
        if extra is not None:
            kl_sum = extra["reduced_qad_kl_sum"] if "reduced_qad_kl_sum" in extra else None
            count = extra["reduced_qad_valid_count"] if "reduced_qad_valid_count" in extra else None
            if isinstance(kl_sum, torch.Tensor) and isinstance(count, torch.Tensor):
                denom = max(float(count.sum().item()), 1.0)
                kl_val = float(kl_sum.sum().item()) / denom
                train_kl_curve.append({"step": step + 1, "train_kl": kl_val})
                cum_kl_curve.append(
                    {"step": step + 1, "cum_mean_kl": sum(c["train_kl"] for c in train_kl_curve) / len(train_kl_curve)}
                )
                # §7.3-style log entry: window-averaged train KL every `train_kl_interval` steps
                # (HF-Trainer log_history semantics — the ModelOpt lane reads first vs last
                # averaged entries, not raw per-row noise).
                kl_window.append(kl_val)
                if len(kl_window) == train_kl_interval or (step + 1) == total_step:
                    kl10_curve.append({"step": step + 1, "kl": sum(kl_window) / len(kl_window)})
                    kl_window.clear()
            total_loss = train_step_info.get("total_loss")
            if isinstance(total_loss, (int, float)):
                loss_curve.append({"step": step + 1, "loss": float(total_loss)})
        if (step + 1) % holdout_interval == 0:
            t0 = time.time()
            kl = trainer.evaluate_holdout(holdout_batches)  # noqa: F821 -- late-bound: trainer defined below, callbacks fire after
            holdout_kl_curve.append({"step": step + 1, "holdout_kl": kl, "eval_s": round(time.time() - t0, 2)})
            trainer.logger.info(f"[effect] holdout KL @ step {step + 1}: {kl:.6f}")  # noqa: F821

    trainer_cfg = TrainerConfig(
        model_cfg=model_cfg,
        optim_cfg=AdamWConfig(
            lr=lr, weight_decay=0.0, betas=(0.9, 0.95), foreach=False, swap_optimizer=True, max_grad_norm=max_grad_norm
        ),
        fsdp_cfg=FSDPConfig(tp_size=1, recompute_ratio=1.0, torch_compile=False),
        dataloader_cfg=DataloaderConfig(
            dataset_config_list=dataset_cfg,
            pack_level="soft",
            pack_max_length=pack_max_length,
            pack_chunk_size=1,
            pack_workers=1,
            global_pack=False,
            num_workers=0,
        ),
        lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0.0, lr_min=lr),
        global_batch_size=global_batch_size,
        total_step=total_steps,
        work_dir=exp_dir / "trainer",
        tokenizer_path=model_dir,
        checkpoint_interval=None,
        hf_interval=None,
        snapshot_interval=None,
        hf_max_keep=None,
        check_health_interval=None,
        exp_tracker="jsonl",
        profile_time=False,
        profile_memory=False,
        hooks_config=HooksConfig(after_train_step=after_step),
    )
    loss_cfg = QADLossConfig(
        mode="qad_topk_forward", top_k=top_k, temperature=1.0, log_prob_min_clamp=-10.0, loss_max_clamp=10.0
    )
    qad_cfg = QADTrainerConfig(
        trainer=trainer_cfg,
        quant=quant_cfg,
        teacher=teacher_cfg,
        loss=loss_cfg,
        model_path=model_dir,
        from_hub="modelscope",
        trainable="full",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    holdout_batches = R.build_holdout_batches(Path(holdout_jsonl), tokenizer, pack_max_length, pad_id, holdout_rows)
    n_holdout_tokens = sum(int(b[0]["seq_ctx"].input_ids.shape[-1]) for b in holdout_batches)

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    # --- Gate A anchor (independent, on the first holdout row's token ids). -----------------
    probe_ids = holdout_batches[0][0]["seq_ctx"].input_ids[0].tolist()
    anchor = gate_a_anchor_independent(model_dir, model_cfg, probe_ids)

    trainer = QADTrainer.from_config(qad_cfg)

    # --- PTQ baseline (fake-quant forward, amax scales, no training) = gate B denominator. --
    t0 = time.time()
    kl_ptq = trainer.evaluate_holdout(holdout_batches)
    ptq_eval_s = time.time() - t0
    holdout_kl_curve.append({"step": 0, "holdout_kl": kl_ptq, "eval_s": round(ptq_eval_s, 2)})
    trainer.logger.info(f"[effect] PTQ baseline holdout KL: {kl_ptq:.6f}")

    # --- ModelOpt-lane premise: master weights snapshot (pre-fit). ---------------------------
    watch_keys = [
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "norm.weight",
    ]
    pre_sd = {k: _full(v) for k, v in trainer._engine.model.state_dict().items()}
    pre_snap = {k: pre_sd[k].float().cpu() for k in watch_keys if k in pre_sd}
    del pre_sd

    # --- QAD run. ---------------------------------------------------------------------------
    t0 = time.time()
    trainer.fit()
    train_time = time.time() - t0

    # --- KL(B_end): end-of-training student in the frozen-scale phase. -----------------------
    kl_b_end = trainer.evaluate_holdout(holdout_batches)
    holdout_kl_curve.append({"step": total_steps, "holdout_kl": kl_b_end, "eval_s": None})

    # --- ModelOpt-lane assertions. ----------------------------------------------------------
    post_sd = {k: _full(v) for k, v in trainer._engine.model.state_dict().items()}
    moved: dict[str, bool | str] = {}
    for k in pre_snap:
        moved[k] = not torch.equal(pre_snap[k], post_sd[k].float().cpu())
    for k in watch_keys:
        if k not in post_sd:
            moved[k] = "absent"
    del post_sd

    losses_all = [c["loss"] for c in loss_curve] + [c["train_kl"] for c in train_kl_curve]
    losses_all += [c["holdout_kl"] for c in holdout_kl_curve] + [c["kl"] for c in kl10_curve]
    all_finite = all(math.isfinite(v) for v in losses_all)
    first_kl = kl10_curve[0]["kl"] if kl10_curve else float("nan")
    final_kl = kl10_curve[-1]["kl"] if kl10_curve else float("nan")
    final_lt_first = bool(final_kl < first_kl)

    # --- Gate C: export -> replay -> bit-exact check. ----------------------------------------
    trainer._engine.optimizer.zero_grad(set_to_none=True)
    export_dir = exp_dir / "export"
    plain = R.clone_student_cpu(model_cfg, trainer, quant_cfg)
    report = export_qad(plain, quant_cfg, export_dir)
    plain_gpu = plain.to(DEVICE)
    replay = R.replay_model_gpu(model_cfg, export_dir, loss_cfg, ema_source=plain_gpu)
    rc = R.holdout_kl_for_model(trainer, replay, holdout_batches, sp_mesh=trainer.sp_mesh)
    kl_c = rc["kl"]

    with torch.no_grad():
        item = dict(holdout_batches[0][0])
        targets = trainer._teacher_targets(item["seq_ctx"], loss_cfg)
        item["teacher_logprobs"] = targets["teacher_logprobs"]
        if "target_token_ids" in targets:
            item["target_token_ids"] = targets["target_token_ids"]
        item["loss_mask"] = _loss_mask(item, loss_cfg)
        ctx_train = trainer._engine.model.build_loss_ctx_batch([item], sp_mesh=trainer.sp_mesh)[0]
        out_train = trainer._engine.model(seq_ctx=item["seq_ctx"].to(DEVICE), loss_ctx={"lm": ctx_train["lm"]})
        logits_train = _full(out_train.logits)
        ctx_replay = replay.build_loss_ctx_batch([item], sp_mesh=None)[0]
        out_replay = replay(seq_ctx=item["seq_ctx"].to(DEVICE), loss_ctx={"lm": ctx_replay["lm"]})
        logits_replay = _full(out_replay.logits)
    logits_equal = bool(torch.equal(logits_train.float(), logits_replay.float()))
    logits_max_abs_diff = float((logits_train.float() - logits_replay.float()).abs().max().item())
    gate_c = bool(logits_equal and math.isfinite(kl_c))
    gate_b = bool(kl_b_end < kl_ptq)

    # --- Save the final master weights (ModelOpt lane: checkpoint saved non-empty). ----------
    master_path = exp_dir / "qad_final_master.safetensors"
    from safetensors.torch import save_file

    sd = {k: _full(v).cpu().contiguous() for k, v in trainer._engine.model.state_dict().items()}
    save_file(sd, str(master_path))
    del sd
    del trainer, plain_gpu, plain
    torch.cuda.empty_cache()

    peak_mem_gib = torch.cuda.max_memory_allocated() / 2**30

    result: dict[str, Any] = {
        "experiment": f"{schema}-kv{kv_dtype}",
        "model_dir": model_dir,
        "model": {
            "num_layers": model_cfg.num_hidden_layers,
            "hidden_size": model_cfg.hidden_size,
            "vocab_size": model_cfg.vocab_size,
            "tie_word_embeddings": bool(model_cfg.tie_word_embeddings),
            "qk_norm": bool(getattr(model_cfg.attention, "qk_norm", False)),
        },
        "quant": quant_cfg.model_dump(),
        "steps": total_steps,
        "lr": lr,
        "global_batch_size": global_batch_size,
        "max_grad_norm": max_grad_norm,
        "loss_mode": "qad_topk_forward",
        "top_k": top_k,
        "teacher_offload": teacher_offload,
        "gate_a_anchor": anchor,
        "kl_ptq": kl_ptq,
        "kl_b_end": kl_b_end,
        "kl_c": kl_c,
        "kl_c_abs_diff": abs(kl_c - kl_b_end),
        "kl_curve_holdout": holdout_kl_curve,
        "train_kl_curve": train_kl_curve,
        "train_kl_window_curve": kl10_curve,
        "cum_mean_kl_curve": cum_kl_curve,
        "loss_curve": loss_curve,
        "holdout": {"rows": len(holdout_batches), "tokens_per_eval": n_holdout_tokens, "jsonl": holdout_jsonl},
        "modelopt_lane": {
            "final_loss < first_loss": final_lt_first,
            "first_train_kl": first_kl,
            "final_train_kl": final_kl,
            "first_train_kl_is_window_mean": True,
            "first_cum_mean_kl": cum_kl_curve[0]["cum_mean_kl"] if cum_kl_curve else None,
            "final_cum_mean_kl": cum_kl_curve[-1]["cum_mean_kl"] if cum_kl_curve else None,
            "final_lt_first_cum_mean": bool(
                cum_kl_curve and cum_kl_curve[-1]["cum_mean_kl"] < cum_kl_curve[0]["cum_mean_kl"]
            ),
            "first_loss_total": loss_curve[0]["loss"] if loss_curve else None,
            "final_loss_total": loss_curve[-1]["loss"] if loss_curve else None,
            "all_finite": all_finite,
            "master_weights_moved": moved,
            "master_checkpoint": {"path": str(master_path), "bytes": master_path.stat().st_size},
            "n_train_kl_logged": len(train_kl_curve),
        },
        "gate_b": gate_b,
        "gate_c": gate_c,
        "gate_c_logits_equal": logits_equal,
        "gate_c_logits_max_abs_diff": logits_max_abs_diff,
        "export": {k: report[k] for k in ("weights_path", "metadata_path", "num_modules", "num_kv_handles")},
        "env": {
            "XTUNER_HF_IMPL": os.environ.get("XTUNER_HF_IMPL", ""),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        },
        "timing_s": {
            "train": round(train_time, 1),
            "ptq_eval": round(ptq_eval_s, 2),
            "total": round(time.time() - t_start, 1),
            "per_step_avg": round(train_time / max(total_steps, 1), 3),
        },
        "peak_gpu_mem_gib": round(peak_mem_gib, 2),
    }

    if generation:
        from generation_probe import build_generation_probe  # noqa: PLC0415

        gen = build_generation_probe(
            model_cfg=model_cfg,
            model_dir=model_dir,
            tokenizer=tokenizer,
            quant_cfg=quant_cfg,
            qad_model=replay,
            max_gen_tokens=max_gen_tokens,
            out_dir=exp_dir,
        )
        result["generation"] = gen
        del replay
        torch.cuda.empty_cache()
    else:
        del replay

    (exp_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[effect] DONE {schema}-kv{kv_dtype}: PTQ={kl_ptq:.6f} B_end={kl_b_end:.6f} C={kl_c:.6f} "
        f"gate_b={gate_b} gate_c={gate_c} final<first={final_lt_first} finite={all_finite} "
        f"train_s={train_time:.0f} peak={peak_mem_gib:.2f}GiB",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--schema", default="w8a8")
    parser.add_argument("--kv", default="int8")
    parser.add_argument("--total-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-kl-interval", type=int, default=10)
    parser.add_argument("--holdout-interval", type=int, default=50)
    parser.add_argument("--pack-max-length", type=int, default=1024)
    parser.add_argument("--teacher-offload", type=int, default=0, help="0: teacher resident on GPU (contract)")
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--max-gen-tokens", type=int, default=96)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    run_qad(
        model_dir=args.model_dir,
        train_jsonl=args.train_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        exp_dir=args.exp_dir,
        schema=args.schema,
        kv_dtype=args.kv,
        total_steps=args.total_steps,
        lr=args.lr,
        train_kl_interval=args.train_kl_interval,
        holdout_interval=args.holdout_interval,
        pack_max_length=args.pack_max_length,
        teacher_offload=bool(args.teacher_offload),
        generation=not args.no_generation,
        max_gen_tokens=args.max_gen_tokens,
        global_batch_size=args.global_batch_size,
        max_grad_norm=args.max_grad_norm,
    )


if __name__ == "__main__":
    main()
