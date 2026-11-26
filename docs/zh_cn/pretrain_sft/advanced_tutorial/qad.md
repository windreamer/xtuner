# 量化感知蒸馏（QAD）

## 概述

量化感知蒸馏（Quantization-Aware Distillation）在 PTQ 之后恢复量化模型的精度：student 是 fake-quant 模型（W8A8/W4A8，激活 per-token/per-batch 动态量化；KV cache int8 以 quantize-dequantize 模拟），teacher 是量化前的 BF16 原模型（frozen），训练目标是两者输出分布的 forward KL divergence（默认 temperature T=1）。与 QAT 不同，QAD 不复制原训练流程，仅用远小于原 post-training 的数据量（论文报告 0.3B–6B tokens）使 student 分布贴合 teacher，因此适用于 SFT/RL/merge 之后的任意多阶段 post-training 模型。参考论文：[Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery](https://arxiv.org/abs/2601.20088)（arXiv 2601.20088）。

XTuner 的实现位于：

| 模块 | 内容 |
| --- | --- |
| `xtuner/v1/quant/` | 量化 schema 配置、IntQuantizer、FakeQuantLinear、KVCache 模拟、QAD loss、teacher adapter |
| `xtuner/v1/train/qad_trainer.py` | `QADTrainer(Trainer)` 与 `QADTrainerConfig` |
| `recipe/qad/` | GLM-5.3-Flash 减层模型配置与启动脚本 |

核心 trainer（`xtuner/v1/train/trainer.py`）零改动：量化在引擎构建后应用（`state_dict` key 不变），teacher targets 在每个 optimizer step 的 `_prepare_model_input` 中注入 loss context。

## 数值契约

fake-quant 的插入位置、granularity、scale 计算与 rounding（roundTiesToEven、`eps_fp32` 下限、zero 包含、clamp 边界）与 msmodelslim 部署 schema 逐条对齐，包括 W4A8 activation 沿 `dim=0`（batch 维）统计的部署特例。契约以设计文档 §4.3 为唯一标准，本文不再重复。要点：

- W8A8：weight per-channel int8，activation per-token dynamic int8。
- W4A8：weight int4 per-group（`group_size=64`，要求 `in_features % group_size == 0`）或 per-channel，activation per-batch dynamic int8。
- KV cache：仅 int8，per-(KV head, head_dim) channel，统计沿 batch·seq 维归约；bf16 KV（部署默认态，"KVCache 16"）通过 `kv_cache=None` 表达，不存在 int16 KV 量化路径。
- scale 无 learnable 参数：weight scale 每 optimizer step 重算，KV amax 以 EMA 平滑（`kv_amax_ema_decay`）；训练末 `scale_freeze_last_ratio`（默认 0.1）比例的步数内全部 scale 定格，导出 metadata 与训练末期模拟一致（设计文档 §5.3、§9.2 的 KL 链路前提）。
- loss：`qad_topk_forward`（teacher top-K ids + logprobs，复用 OPD KL kernel，含 tail bin）或 `qad_full_kl`（全词表）；reduction 走 loss 体系的全局校准（dp/sp/grad-accum 等价）。

## 配置

`QADTrainerConfig` 组合 `TrainerConfig`（模型、优化器、数据、FSDP、checkpoint 原样复用）与 QAD 专属字段：

```python
from xtuner.v1.quant.config import KVQConfig, QADQuantConfig, SelectiveQuantConfig
from xtuner.v1.quant.loss import QADLossConfig
from xtuner.v1.quant.teacher import QADTeacherAdapterConfig
from xtuner.v1.rl.distillation.config import TrainTeacherConfig
from xtuner.v1.train.qad_trainer import QADTrainerConfig

trainer = QADTrainerConfig(
    trainer=trainer_cfg,  # 原样复用的 TrainerConfig
    quant=QADQuantConfig(
        quant_schema="w8a8",  # 或 "w4a8"（w4_granularity、group_size 生效）
        selective=SelectiveQuantConfig(exclude=["lm_head"]),
        kv_cache=KVQConfig(dtype="int8"),  # None = KV cache bf16（对照组）
        kv_amax_ema_decay=0.999,
        scale_freeze_last_ratio=0.1,
    ),
    teacher=QADTeacherAdapterConfig(
        teacher=TrainTeacherConfig(
            name="teacher",
            model_path=teacher_model_path,  # 量化前 BF16 原模型
            model_cfg=teacher_model_cfg,
            fsdp_cfg=None,
        ),
        teacher_offload=False,  # True 时按 onload -> forward -> offload 节奏执行
        chunk_size=512,
    ),
    loss=QADLossConfig(mode="qad_topk_forward", top_k=64, temperature=1.0),
    model_path=model_path,
    from_hub="modelscope",  # repo id 经 ModelScope 解析；XTUNER_FROM_HUB 可回退覆盖
    trainable="full",
    holdout=None,  # holdout 评估间隔，见下文"限制"
)
```

loss mode 两种契约：`qad_topk_forward` 要求 teacher 产出 `target_token_ids [B,T,K] + teacher_logprobs [B,T,K]`（Training-side 与 Rollout teacher 均可）；`qad_full_kl` 消费 `teacher_logprobs [B,T,V]`（仅 Training-side，chunked 传输不整块驻留）。训练指标：`qad/train_kl`、`qad/topk_overlap_fraction`（top-K 模式）、`qad/teacher_time_{compute,onload,offload}`。

数据直接复用 SFT packed 管线，不要求额外标注：`loss_mask` 由 batch 的 `shifted_labels` 对 ignore index 取反导出。teacher 前向按 packed 段边界（`cu_seq_lens`）逐段执行，与 student 的 varlen attention 保持相同的上下文边界。

## 一键运行

```shell
DATA_PATH=/path/to/sft.jsonl \
GLM53_FLASH_MODEL_PATH=zai-org/GLM-5.3-Flash \
NPROC_PER_NODE=8 \
bash recipe/qad/scripts/run_qad.sh
```

脚本导出 `XTUNER_FROM_HUB=modelscope`；`GLM53_FLASH_MODEL_PATH` 为本地截断模型目录时不受该设置影响。可选环境变量：`TEACHER_MODEL_PATH`（默认取 student 路径，即同权重 BF16 副本）、`QAD_SCHEMA`（`w8a8`/`w4a8`）、`TEACHER_OFFLOAD`、`TOTAL_STEP`、`GLOBAL_BATCH_SIZE`、`PACK_MAX_LENGTH`、`QAD_LR`（默认 1e-5，论文 §4.2 建议区间 1e-6..1e-5）。

## 限制

- **KV cache 只有 int8**。不存在 int16 KV 量化路径：int8 是模拟对象，bf16（`kv_cache=None`）是部署默认态的对照组。
- **LoRA 未接入**。`trainable="lora"` 在配置解析阶段抛出 `NotImplementedError`：基础引擎在 `TrainEngine.__init__` 内即用全量模型参数构建 optimizer，adapter-only 参数组需要改动引擎才能表达。`FakeQuantLinear` 的 merge-then-quant（`delta_w` 契约）已就绪，LoRA 组合留待引擎侧支持。当前唯一训练路径是 `trainable="full"`。
- **holdout 评估未接入训练循环**。SFT `fit()` 没有评估缝；`HoldoutConfig.interval` 仅做配置校验，`QADTrainer.evaluate_holdout(data_batches)` 供训练 hook 调用并返回 holdout KL（`qad/holdout_kl` 的 checkpoint 选点指标），不会自动在循环内触发。
- **lm_head 不在默认量化集合内**（设计 §4.1）：它是输出分布的直接来源，且其 forward 契约与 fake-quant 单参数签名不同，配置必须通过 `selective.exclude=["lm_head"]` 排除。
- 部署侧产物（`qad_quant_metadata.json` + msmodelslim convert）由 export 任务提供，见设计文档 §9.1。
