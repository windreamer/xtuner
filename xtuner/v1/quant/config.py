from typing import Any, Literal

from pydantic import BaseModel, Field


class SelectiveQuantConfig(BaseModel):
    """Layer selection config for fake-quant insertion, controlled by fnmatch
    module-name patterns.

    Args:
        include (list[str]): fnmatch patterns matched against module names; matched modules are quantized.
        exclude (list[str]): fnmatch patterns whose matches are kept in bf16; takes precedence over ``include``.
        exclude_first_k_blocks (int): Number of leading decoder blocks to keep unquantized.
        exclude_last_k_blocks (int): Number of trailing decoder blocks to keep unquantized.
    """

    model_config = {"extra": "forbid"}

    include: list[str] = ["*"]
    exclude: list[str] = []
    exclude_first_k_blocks: int = Field(default=0, ge=0)
    exclude_last_k_blocks: int = Field(default=0, ge=0)


class KVQConfig(BaseModel):
    """KV cache quantization config; only int8 is supported in v1.

    Args:
        dtype (Literal["int8"]): Quantized storage dtype of the KV cache. KV cache "16" (bf16) is
            expressed as ``kv_cache=None`` on :class:`QADQuantConfig`, not by this config.
    """

    model_config = {"extra": "forbid"}

    dtype: Literal["int8"] = "int8"


class QADQuantConfig(BaseModel):
    """Quantization schema config for QAD training.

    Args:
        schema (Literal["w8a8", "w4a8"]): Weight-activation quantization schema.
        w4_granularity (Literal["per_group", "per_channel"]): Weight granularity for ``w4a8``.
        group_size (int): Group size for ``per_group`` weight quantization; must be positive.
        selective (SelectiveQuantConfig): Layer selection policy.
        kv_cache (KVQConfig | None): KV cache quantization; ``None`` means the KV cache stays bf16.
        kv_amax_ema_decay (float): EMA decay for KV amax smoothing, in ``[0, 1)``.
        scale_freeze_last_ratio (float): Fraction of trailing steps with scales frozen, in ``[0, 1]``.
    """

    model_config = {"extra": "forbid"}

    quant_schema: Literal["w8a8", "w4a8"] = "w8a8"
    w4_granularity: Literal["per_group", "per_channel"] = "per_group"
    group_size: int = Field(default=64, gt=0)
    selective: SelectiveQuantConfig = Field(default_factory=SelectiveQuantConfig)
    kv_cache: KVQConfig | None = None
    kv_amax_ema_decay: float = Field(default=0.999, ge=0, lt=1)
    scale_freeze_last_ratio: float = Field(default=0.1, ge=0, le=1)

    @classmethod
    def for_schema(cls, schema: Literal["w8a8", "w4a8"], **kwargs: Any) -> "QADQuantConfig":
        """Build a config with the default layer-exclusion policy of the given
        schema.

        Args:
            schema (Literal["w8a8", "w4a8"]): Target quantization schema.
            **kwargs (Any): Field overrides forwarded to the constructor. When ``selective`` is not
                provided, the schema default policy is applied (see Returns); when provided, it is
                respected as-is and no defaults are merged into it.

        Returns:
            QADQuantConfig: Defaults per design §4.1: ``lm_head`` is always excluded (quantizing it
            directly shifts the output distribution); ``w4a8`` additionally excludes the first and
            last decoder block, ``w8a8`` excludes none.
        """
        first_k, last_k = (1, 1) if schema == "w4a8" else (0, 0)
        if "selective" not in kwargs:
            kwargs["selective"] = SelectiveQuantConfig(
                exclude=["lm_head"], exclude_first_k_blocks=first_k, exclude_last_k_blocks=last_k
            )
        return cls(quant_schema=schema, **kwargs)
