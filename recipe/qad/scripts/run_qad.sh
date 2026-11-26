#!/usr/bin/env bash
set -euo pipefail

# Launch QAD (quantization-aware distillation) training on the GLM-5.3-Flash truncated model.
#
# Required env:
#   DATA_PATH                 jsonl SFT corpus used as distillation data (no label quality needed)
#   GLM53_FLASH_MODEL_PATH    student checkpoint (local path or ModelScope repo id);
#                             defaults to the ModelScope repo id below
#
# Optional env: TEACHER_MODEL_PATH (defaults to the student checkpoint = same-weights BF16 copy),
# WORK_DIR, TOTAL_STEP, GLOBAL_BATCH_SIZE, PACK_MAX_LENGTH, QAD_SCHEMA (w8a8|w4a8),
# TEACHER_OFFLOAD, CHECKPOINT_INTERVAL, NPROC_PER_NODE, NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

: "${DATA_PATH:?DATA_PATH is required}"
export GLM53_FLASH_MODEL_PATH="${GLM53_FLASH_MODEL_PATH:-zai-org/GLM-5.3-Flash}"
# Model weights resolve through ModelScope (design doc §5.7); a local path is unaffected by this.
export XTUNER_FROM_HUB="${XTUNER_FROM_HUB:-modelscope}"
export WORK_DIR="${WORK_DIR:-${REPO_ROOT}/work_dirs/glm53_flash_trunc_qad}"
export QAD_CONFIG_PATH="${REPO_ROOT}/recipe/qad/config/glm53_flash_trunc_qad.py"
export PYTHONUNBUFFERED=1

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
torchrun \
  --nproc-per-node="${NPROC_PER_NODE:-8}" \
  --nnodes="${NNODES:-1}" \
  --node-rank="${NODE_RANK:-0}" \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  --master-port="${MASTER_PORT:-6000}" \
  --no_python \
  python -c '
import os

from xtuner.v1.train.qad_trainer import QADTrainer
from xtuner.v1.utils import Config

config = Config.fromfile(os.environ["QAD_CONFIG_PATH"])["trainer"]
trainer = QADTrainer.from_config(config)
trainer.fit()
'
