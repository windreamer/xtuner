import pytest
from pydantic import ValidationError

from xtuner.v1.quant import KVQConfig, QADQuantConfig, SelectiveQuantConfig


class TestDefaults:
    def test_selective_defaults(self) -> None:
        cfg = SelectiveQuantConfig()
        assert cfg.include == ["*"]
        assert cfg.exclude == []
        assert cfg.exclude_first_k_blocks == 0
        assert cfg.exclude_last_k_blocks == 0

    def test_kvq_defaults(self) -> None:
        cfg = KVQConfig()
        assert cfg.dtype == "int8"

    def test_qad_defaults(self) -> None:
        cfg = QADQuantConfig()
        assert cfg.quant_schema == "w8a8"
        assert cfg.w4_granularity == "per_group"
        assert cfg.group_size == 64
        assert cfg.selective == SelectiveQuantConfig()
        assert cfg.kv_cache is None
        assert cfg.kv_amax_ema_decay == 0.999
        assert cfg.scale_freeze_last_ratio == 0.1

    def test_no_symmetric_field(self) -> None:
        assert "symmetric" not in KVQConfig.model_fields


class TestForSchema:
    def test_w4a8_policy(self) -> None:
        cfg = QADQuantConfig.for_schema("w4a8")
        assert cfg.quant_schema == "w4a8"
        assert cfg.selective.exclude == ["lm_head"]
        assert cfg.selective.exclude_first_k_blocks == 1
        assert cfg.selective.exclude_last_k_blocks == 1

    def test_w8a8_policy(self) -> None:
        cfg = QADQuantConfig.for_schema("w8a8")
        assert cfg.quant_schema == "w8a8"
        assert "lm_head" in cfg.selective.exclude
        assert cfg.selective.exclude_first_k_blocks == 0
        assert cfg.selective.exclude_last_k_blocks == 0

    def test_user_selective_respected_as_is(self) -> None:
        cfg = QADQuantConfig.for_schema("w4a8", selective=SelectiveQuantConfig(exclude=["x"]))
        assert cfg.quant_schema == "w4a8"
        assert cfg.selective.exclude == ["x"]
        assert "lm_head" not in cfg.selective.exclude
        assert cfg.selective.exclude_first_k_blocks == 0
        assert cfg.selective.exclude_last_k_blocks == 0

    def test_quant_schema_w4a8_builds(self) -> None:
        cfg = QADQuantConfig(quant_schema="w4a8")
        assert cfg.quant_schema == "w4a8"
        assert cfg.w4_granularity == "per_group"
        assert cfg.group_size == 64

    def test_no_shared_mutation(self) -> None:
        cfg = QADQuantConfig.for_schema("w4a8")
        cfg.selective.exclude_first_k_blocks = 5
        assert QADQuantConfig.for_schema("w4a8").selective.exclude_first_k_blocks == 1


class TestValidation:
    def test_group_size_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QADQuantConfig(group_size=0)

    def test_ema_decay_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QADQuantConfig(kv_amax_ema_decay=1.0)

    def test_freeze_ratio_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QADQuantConfig(scale_freeze_last_ratio=1.5)

    def test_kv_dtype_int16_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KVQConfig(dtype="int16")  # type: ignore[arg-type]

    def test_unknown_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QADQuantConfig(unknown_field=1)  # type: ignore[arg-type]

    def test_schema_literal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QADQuantConfig(schema="w8a16")  # type: ignore[arg-type]
