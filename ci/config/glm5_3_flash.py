import os

from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.glm5_3 import Glm53FlashTowerConfig
from xtuner.v1.train import TrainerConfig


GLM53_FLASH_PATH = os.environ["GLM53_FLASH_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]

# BF16 training on the dequantized (BF16) release. The FP8 release needs the
# float8 pipeline; see docs/design/model/glm_5_3_flash.md.
model_cfg = Glm53FlashTowerConfig(
    ep_size=1,
    dispatcher=None,  # eager grouped-expert path; deepep wiring is a follow-up
)

optim_cfg = AdamWConfig(lr=6e-05)

lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=False,
    cpu_offload=False,
    ep_size=model_cfg.ep_size,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=4096),
    },
]

dataloader_config = DataloaderConfig(pack_max_length=32768)

loss_cfg = CELossConfig(mode="chunk")


trainer = TrainerConfig(
    load_from=GLM53_FLASH_PATH,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=GLM53_FLASH_PATH,
    global_batch_size=16,
    work_dir="work_dirs/glm-5.3-flash/bf16-32k",
    seed=0,
    strict_load=False,
    total_step=1000000,
    profile_step=10,
    intra_layer_micro_batch=1,
)
