#!/usr/bin/env bash
set -euo pipefail

# T9 local simulation acceptance pipeline (goal doc T9, design doc §10.2):
#   probe -> download-or-fallback -> truncate -> synthetic corpus -> 3 QAD micro-runs
#   (W8A8+KV8, W4A8+KV8, W4A8+KV-bf16) -> export -> replay -> gates B/C -> report.md
#
# Model weights resolve through ModelScope ONLY (design doc §1.4). The recipe's default repo
# (zai-org/GLM-5.3-Flash) is probed first; if the probe fails or the weights exceed ~15GB, the
# sanctioned fallback builds the 4-layer truncated model from the REAL config values with
# seeded RANDOM weights (recorded in report.md §1.2).
#
# Env overrides: SIM_WORK_DIR (default work_dir/qad_sim), TOTAL_STEPS, QAD_LR, EVAL_INTERVAL,
# QAD_SIM_SEED, MASTER_PORT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SIM_DIR="${SIM_WORK_DIR:-${REPO_ROOT}/work_dir/qad_sim}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${SIM_DIR}/logs"
mkdir -p "${LOG_DIR}"

export TOTAL_STEPS="${TOTAL_STEPS:-150}"
export QAD_LR="${QAD_LR:-1e-5}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-30}"
export QAD_SIM_SEED="${QAD_SIM_SEED:-42}"
export MASTER_PORT="${MASTER_PORT:-61117}"
export PYTHONUNBUFFERED=1
# flash-attn is not installed in this environment and torch 2.13's compiled flex_attention
# fails on 2048-token packed rows (data-dependent guard, see logs). Force the bitwise
# HF-parity eager attention for teacher, student and replay alike (consistency is what the
# KL chain needs; all three sides share the same eager path).
export XTUNER_HF_IMPL="${XTUNER_HF_IMPL:-true}"

MODEL_DIR="${SIM_DIR}/model/trunc_glm53_4l"

echo "[sim] stage 1/6: ModelScope probe" | tee "${LOG_DIR}/pipeline.log"
cd "${REPO_ROOT}"
"${PYTHON}" work_dir/qad_sim/probe_modelscope.py 2>&1 | tee -a "${LOG_DIR}/pipeline.log"

echo "[sim] stage 2/6: build truncated model (download-or-fallback)" | tee -a "${LOG_DIR}/pipeline.log"
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  (cd work_dir/qad_sim && "${PYTHON}" build_trunc_model.py --model-dir "${MODEL_DIR}" --info-json "${SIM_DIR}/model_info.json" --mode auto) 2>&1 | tee -a "${LOG_DIR}/pipeline.log"
fi
if [[ ! -f "${SIM_DIR}/model_info.json" ]]; then
  cp "${MODEL_DIR}/../model_info.json" "${SIM_DIR}/model_info.json" 2>/dev/null || true
fi

echo "[sim] stage 3/6: synthetic corpus" | tee -a "${LOG_DIR}/pipeline.log"
(cd work_dir/qad_sim && "${PYTHON}" make_corpus.py \
  --train-jsonl "${SIM_DIR}/data/train.jsonl" \
  --holdout-jsonl "${SIM_DIR}/data/holdout.jsonl" \
  --num-train 6500 --num-holdout 40 --seed 1234) 2>&1 | tee -a "${LOG_DIR}/pipeline.log"
"${PYTHON}" - "$SIM_DIR" <<'EOF' 2>&1 | tee -a "${LOG_DIR}/pipeline.log"
import json, sys
from pathlib import Path
wd = Path(sys.argv[1])
def count(p: Path) -> int:
    return sum(1 for _ in p.open(encoding="utf-8"))
info = {
    "train_jsonl": str(wd / "data/train.jsonl"),
    "holdout_jsonl": str(wd / "data/holdout.jsonl"),
    "num_train": count(wd / "data/train.jsonl"),
    "num_holdout": count(wd / "data/holdout.jsonl"),
    "seed": 1234,
    "note": "deterministic synthetic corpus (no local jsonl corpus available); tokenized with the local Qwen3-0.6B tokenizer through the xtuner SFT pipeline (intern-s1 chat template)",
}
(wd / "corpus_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print("[corpus]", info)
EOF

echo "[sim] stage 4/6: experiments (LoRA probe + W8A8+KV8 + W4A8+KV8 + W4A8+KV-bf16)" | tee -a "${LOG_DIR}/pipeline.log"
PYTHONPATH="${REPO_ROOT}/work_dir/qad_sim${PYTHONPATH:+:${PYTHONPATH}}" \
torchrun --nproc-per-node=1 --nnodes=1 --master-addr=127.0.0.1 --master-port="${MASTER_PORT}" \
  work_dir/qad_sim/run_all.py \
  --model-dir "${MODEL_DIR}" \
  --train-jsonl "${SIM_DIR}/data/train.jsonl" \
  --holdout-jsonl "${SIM_DIR}/data/holdout.jsonl" \
  --work-dir "${SIM_DIR}" \
  --total-steps "${TOTAL_STEPS}" \
  --lr "${QAD_LR}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --pack-max-length 1024 2>&1 | tee -a "${LOG_DIR}/pipeline.log"

echo "[sim] stage 5/6: export + replay already executed per experiment inside stage 4" | tee -a "${LOG_DIR}/pipeline.log"

echo "[sim] stage 6/6: report" | tee -a "${LOG_DIR}/pipeline.log"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON}" work_dir/qad_sim/make_report.py --work-dir "${SIM_DIR}" 2>&1 | tee -a "${LOG_DIR}/pipeline.log"

echo "[sim] DONE: ${SIM_DIR}/report.md" | tee -a "${LOG_DIR}/pipeline.log"
