# Effect-lane: greedy-generation comparison across BF16 / PTQ / QAD(replay) on the fixed
# 30-prompt probe set. All three variants share the same xtuner forward path (same eager
# attention, same tokenizer/template) so the only difference is the quantization state:
#   bf16: plain model, checkpoint weights as-is (== the QAD teacher)
#   ptq : fake-quant W8A8 + KV int8 with amax scales, untrained (== the QAD student init)
#   qad : export_qad/load_qad_export replay of the trained student (deployment artifact)
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.quant.fake_quant_linear import apply_quantization
from xtuner.v1.quant.kv_cache import apply_kv_quantization  # norm-safe patched by run_real
from xtuner.v1.utils.device import get_device


DEVICE = get_device()

_PROMPTS = None


def _prompts() -> list[dict]:
    global _PROMPTS
    if _PROMPTS is None:
        from probe_prompts import PROMPTS

        _PROMPTS = PROMPTS
    return _PROMPTS


def _full(t: torch.Tensor | Any) -> torch.Tensor:
    return t.full_tensor() if isinstance(t, DTensor) else t.detach().clone()


def _seq_ctx(ids: torch.Tensor) -> SequenceContext:
    t_len = int(ids.shape[1])
    return SequenceContext(
        input_ids=ids,
        cu_seq_lens_q=torch.tensor([0, t_len], dtype=torch.int),
        cu_seq_lens_k=torch.tensor([0, t_len], dtype=torch.int),
        max_length_q=t_len,
        max_length_k=t_len,
        device=str(DEVICE),
    ).to(DEVICE)


def _plain_model(model_cfg: Any, model_dir: str) -> Any:
    with torch.device(DEVICE):
        model = model_cfg.model_copy(deep=True).build()
    _loaded, _unloaded, missing = model.from_hf(model_dir, strict=False)
    assert not missing, f"plain model load: missing keys {sorted(missing)[:5]}"
    return model


@torch.no_grad()
def _greedy(model: nn.Module, prompt_ids: list[int], eos_id: int, max_new: int) -> list[int]:
    ids = list(prompt_ids)
    for _ in range(max_new):
        t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        out = model(seq_ctx=_seq_ctx(t), loss_ctx=None)
        logits = out.logits
        if isinstance(logits, tuple):
            logits = logits[0]
        nxt = int(_full(logits)[0, -1].argmax().item())
        ids.append(nxt)
        if nxt == eos_id:
            break
    return ids


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _correct(output: str, checks: list[str] | None) -> bool | None:
    if not checks:
        return None
    low = _norm(output)
    return any(c in low for c in checks)


def build_generation_probe(
    model_cfg: Any,
    model_dir: str,
    tokenizer: Any,
    quant_cfg: Any,
    qad_model: nn.Module,
    max_gen_tokens: int = 96,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    eos_id = tokenizer.eos_token_id

    def _prompt_ids(prompt: str) -> list[int]:
        """Chat-template the probe prompt (Qwen3 native template, thinking disabled)."""
        try:
            tokenized = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, enable_thinking=False
            )
        except TypeError:  # template without the enable_thinking variable
            tokenized = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
            )
        ids = tokenized["input_ids"] if "input_ids" in tokenized else tokenized
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    variants: dict[str, nn.Module] = {"bf16": _plain_model(model_cfg, model_dir)}
    ptq = _plain_model(model_cfg, model_dir)
    apply_quantization(ptq, quant_cfg)
    apply_kv_quantization(ptq, quant_cfg.kv_cache, quant_cfg.kv_amax_ema_decay)
    variants["ptq"] = ptq
    variants["qad"] = qad_model

    per_variant: dict[str, Any] = {}
    try:
        for name, model in variants.items():
            t0 = time.time()
            rows = []
            for p in _prompts():
                pids = _prompt_ids(p["prompt"])
                gen_ids = _greedy(model, pids, eos_id, max_gen_tokens)
                new_ids = gen_ids[len(pids) :]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                rows.append(
                    {
                        "id": p["id"],
                        "kind": p["kind"],
                        "prompt": p["prompt"],
                        "output": text,
                        "n_gen_tokens": len(new_ids),
                        "correct": _correct(text, p["check"]),
                    }
                )
            per_variant[name] = {
                "outputs": rows,
                "gen_time_s": round(time.time() - t0, 1),
                "n_correct": sum(1 for r in rows if r["correct"] is True),
                "n_checkable": sum(1 for r in rows if r["correct"] is not None),
            }
    finally:
        variants["bf16"].cpu()
        variants["ptq"].cpu()
        del variants
        torch.cuda.empty_cache()

    # Pairwise exact-match agreement on normalized full outputs.
    agreement: dict[str, float] = {}
    texts = {name: [_norm(r["output"]) for r in per_variant[name]["outputs"]] for name in per_variant}
    for a, b in (("bf16", "ptq"), ("bf16", "qad"), ("ptq", "qad")):
        n = len(texts[a])
        agreement[f"{a}_vs_{b}"] = sum(1 for x, y in zip(texts[a], texts[b]) if x == y) / n

    # First-token agreement (greedy next-token identity) as a finer signal.
    first_tok = {
        name: [(_norm(r["output"]).split(" ")[0] if r["output"] else "") for r in per_variant[name]["outputs"]]
        for name in per_variant
    }
    first_tok_agreement: dict[str, float] = {}
    for a, b in (("bf16", "ptq"), ("bf16", "qad"), ("ptq", "qad")):
        n = len(first_tok[a])
        first_tok_agreement[f"{a}_vs_{b}"] = sum(1 for x, y in zip(first_tok[a], first_tok[b]) if x == y) / n

    correctness = {name: per_variant[name]["n_correct"] for name in per_variant}
    result = {
        "prompt_source": "work_dir/qad_sim/effect/probe_prompts.py (30 authored prompts)",
        "chat_template": "qwen3 (tokenizer native, enable_thinking=False)",
        "max_gen_tokens": max_gen_tokens,
        "decoding": "greedy argmax, eos stop",
        "agreement_exact_match": agreement,
        "agreement_first_word": first_tok_agreement,
        "correctness_counts": correctness,
        "n_checkable": per_variant["bf16"]["n_checkable"],
        "gen_time_s": {name: per_variant[name]["gen_time_s"] for name in per_variant},
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "generation_outputs.json").write_text(
            json.dumps(per_variant, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return result
