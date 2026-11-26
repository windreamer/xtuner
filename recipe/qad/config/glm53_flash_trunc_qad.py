# QAD (quantization-aware distillation) recipe for the GLM-5.3-Flash truncated model.
#
# Design doc: /data/windreamer/codebase/qad_ascend_design.md §8. The trainer subclass and config
# live in xtuner/v1/train/qad_trainer.py; launch via recipe/qad/scripts/run_qad.sh.
import os
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.sft_tokenize_fn import OpenaiTokenizeFunctionConfig
from xtuner.v1.model import Qwen3DenseConfig, get_model_config_from_hf
from xtuner.v1.quant.config import KVQConfig, QADQuantConfig, SelectiveQuantConfig
from xtuner.v1.quant.loss import QADLossConfig
from xtuner.v1.quant.teacher import QADTeacherAdapterConfig
from xtuner.v1.rl.distillation.config import TrainTeacherConfig
from xtuner.v1.train.qad_trainer import QADTrainerConfig
from xtuner.v1.train.trainer import TrainerConfig


work_dir = os.environ.get("WORK_DIR", "work_dirs/glm53_flash_trunc_qad")
# GLM-5.3-Flash truncated checkpoint (e.g. a locally truncated copy of the ModelScope repo).
model_path = os.environ.get("GLM53_FLASH_MODEL_PATH", "zai-org/GLM-5.3-Flash")
data_path = os.environ.get("DATA_PATH")
if data_path is None:
    raise RuntimeError("DATA_PATH is required (jsonl SFT corpus for distillation)")

total_train_steps = int(os.environ.get("TOTAL_STEP", "1000"))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "16"))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", "8192"))
# QAD is a precision-recovery phase; the paper §4.2 range is 1e-6..1e-5 (LR-heavy models may use
# the upper end). Constant schedule, no warmup.
lr = float(os.environ.get("QAD_LR", "1e-5"))

model_cfg: Qwen3DenseConfig = get_model_config_from_hf(Path(model_path))
model_cfg.compile_cfg = False

fsdp_cfg = FSDPConfig(tp_size=1, recompute_ratio=1.0, torch_compile=False)

# 1. Quantization: W8A8 dynamic + KV-cache int8 simulation (design §4.2/§6). "KVCache 16" (bf16)
#    is the un-quantized control: set kv_cache=None. lm_head/embedding stay bf16 (design §4.1).
quant_cfg = QADQuantConfig(
    quant_schema=os.environ.get("QAD_SCHEMA", "w8a8"),
    w4_granularity="per_group",
    group_size=64,
    selective=SelectiveQuantConfig(exclude=["lm_head"]),
    kv_cache=KVQConfig(dtype="int8"),
    kv_amax_ema_decay=0.999,
    scale_freeze_last_ratio=0.1,
)

# 2. Teacher: same-architecture BF16 frozen copy (pre-quantization weights), training-side.
teacher_cfg = QADTeacherAdapterConfig(
    teacher=TrainTeacherConfig(
        name="teacher",
        model_path=os.environ.get("TEACHER_MODEL_PATH", model_path),
        model_cfg=get_model_config_from_hf(Path(os.environ.get("TEACHER_MODEL_PATH", model_path))),
        fsdp_cfg=FSDPConfig(tp_size=1, recompute_ratio=0, torch_compile=False, requires_grad=False),
    ),
    teacher_offload=bool(int(os.environ.get("TEACHER_OFFLOAD", "0"))),
    chunk_size=512,
)

loss_cfg = QADLossConfig(
    mode="qad_topk_forward",
    top_k=64,
    temperature=1.0,
    log_prob_min_clamp=-10.0,
    loss_max_clamp=10.0,
)

dataset_cfg = [
    {
        "dataset": DatasetConfig(name="qad_sft", anno_path=data_path, sample_ratio=1.0),
        "tokenize_fn": OpenaiTokenizeFunctionConfig(chat_template="glm5.3", max_length=pack_max_length),
    }
]
dataloader_cfg = DataloaderConfig(
    dataset_config_list=dataset_cfg,
    pack_level="soft",
    pack_max_length=pack_max_length,
    pack_chunk_size=1,
    pack_workers=1,
    global_pack=False,
    num_workers=0,
)

optim_cfg = AdamWConfig(lr=lr, weight_decay=0.0, betas=(0.9, 0.95), foreach=False)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0.0, lr_min=lr)

trainer_cfg = TrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataloader_cfg=dataloader_cfg,
    lr_cfg=lr_cfg,
    global_batch_size=global_batch_size,
    total_step=total_train_steps,
    work_dir=work_dir,
    tokenizer_path=os.environ.get("TOKENIZER_PATH", model_path),
    checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", "200")),
    hf_interval=None,
    snapshot_interval=None,
    exp_tracker="jsonl",
    profile_time=False,
    profile_memory=False,
)

trainer = QADTrainerConfig(
    trainer=trainer_cfg,
    quant=quant_cfg,
    teacher=teacher_cfg,
    loss=loss_cfg,
    model_path=model_path,
    from_hub="modelscope",
    trainable="full",
    holdout=None,
)
