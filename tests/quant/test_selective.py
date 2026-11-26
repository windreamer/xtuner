"""Unit tests for selective quantization policy (design doc §4.1, §5.4, §10.1
TestSelectiveQuant)."""

import pytest
from torch import nn

from xtuner.v1.quant.config import QADQuantConfig, SelectiveQuantConfig
from xtuner.v1.quant.fake_quant_linear import FakeQuantLinear, apply_quantization, remove_quantization


def build_dummy_model(num_blocks: int = 6) -> nn.Module:
    """Dummy decoder model named model.layers.{i}.mlp.gate_proj / down_proj +
    embedding + lm_head."""

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Sequential()
            self.mlp.gate_proj = nn.Linear(8, 16, bias=False)  # type: ignore[attr-defined]
            self.mlp.down_proj = nn.Linear(16, 8, bias=False)  # type: ignore[attr-defined]

    class Dummy(nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 8)
            self.layers = nn.ModuleList(Block() for _ in range(n))
            self.lm_head = nn.Linear(8, 32, bias=False)

    return Dummy(num_blocks)


def quantized_names(model: nn.Module) -> set[str]:
    return {name for name, m in model.named_modules() if isinstance(m, FakeQuantLinear)}


class TestSelectiveQuant:
    def test_include_default_all_linear(self) -> None:
        model = build_dummy_model()
        apply_quantization(model, QADQuantConfig(quant_schema="w8a8"))
        names = quantized_names(model)
        assert len(names) == 13  # 6 blocks x 2 + lm_head
        assert "lm_head" in names
        # only nn.Linear instances change; embedding is untouched
        assert not isinstance(model.embed_tokens, nn.Linear)

    def test_exclude_takes_precedence(self) -> None:
        model = build_dummy_model()
        cfg = QADQuantConfig(
            quant_schema="w8a8",
            selective=SelectiveQuantConfig(include=["*"], exclude=["*.down_proj", "lm_head"]),
        )
        apply_quantization(model, cfg)
        names = quantized_names(model)
        assert names == {f"layers.{i}.mlp.gate_proj" for i in range(6)}
        assert not isinstance(model.lm_head, FakeQuantLinear)

    def test_first_k_blocks_excluded(self) -> None:
        model = build_dummy_model()
        cfg = QADQuantConfig(quant_schema="w8a8", selective=SelectiveQuantConfig(exclude_first_k_blocks=2))
        apply_quantization(model, cfg)
        names = quantized_names(model)
        assert "layers.0.mlp.gate_proj" not in names
        assert "layers.1.mlp.down_proj" not in names
        assert "layers.2.mlp.gate_proj" in names
        assert len(names) == 9  # 4 remaining blocks x 2 + lm_head

    def test_last_k_blocks_excluded(self) -> None:
        model = build_dummy_model()
        cfg = QADQuantConfig(quant_schema="w8a8", selective=SelectiveQuantConfig(exclude_last_k_blocks=1))
        apply_quantization(model, cfg)
        names = quantized_names(model)
        assert "layers.5.mlp.gate_proj" not in names
        assert "layers.4.mlp.down_proj" in names

    def test_first_and_last_default_w4a8_policy(self) -> None:
        model = build_dummy_model()
        apply_quantization(model, QADQuantConfig.for_schema("w4a8"))
        names = quantized_names(model)
        assert "layers.0.mlp.gate_proj" not in names
        assert "layers.5.mlp.down_proj" not in names
        assert "layers.3.mlp.gate_proj" in names

    def test_embedding_and_lm_head_untouched_by_contract(self) -> None:
        # Contract (design doc §4.1): apply_quantization only touches nn.Linear instances;
        # embedding stays bf16 because it is an nn.Embedding, and lm_head exclusion is
        # expressed via cfg.selective.exclude patterns in recipes, not hardcoded here.
        model = build_dummy_model()
        cfg = QADQuantConfig(quant_schema="w8a8", selective=SelectiveQuantConfig(exclude=["lm_head"]))
        apply_quantization(model, cfg)
        assert not isinstance(model.lm_head, FakeQuantLinear)
        assert isinstance(model.lm_head, nn.Linear)
        assert isinstance(model.embed_tokens, nn.Embedding)
        # quantized modules remain nn.Linear subclasses
        for _, m in model.named_modules():
            assert not isinstance(m, FakeQuantLinear) or isinstance(m, nn.Linear)

    def test_remove_restores_all(self) -> None:
        model = build_dummy_model()
        apply_quantization(model, QADQuantConfig(quant_schema="w8a8"))
        remove_quantization(model)
        assert quantized_names(model) == set()
        assert type(model.lm_head) is nn.Linear  # type: ignore[comparison-overlap]
        assert type(model.layers[2].mlp.gate_proj) is nn.Linear

    def test_state_dict_keys_preserved_by_selective_apply(self) -> None:
        model = build_dummy_model()
        keys_before = set(model.state_dict().keys())
        apply_quantization(model, QADQuantConfig(quant_schema="w8a8"))
        assert set(model.state_dict().keys()) == keys_before

    def test_apply_twice_on_model_raises(self) -> None:
        model = build_dummy_model()
        apply_quantization(model, QADQuantConfig(quant_schema="w8a8"))
        with pytest.raises(RuntimeError, match="already quantized"):
            apply_quantization(model, QADQuantConfig(quant_schema="w8a8"))
