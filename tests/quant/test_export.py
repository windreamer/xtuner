"""Tests for QAD export + replay (design doc §9.1, §10.1 TestExport)."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from xtuner.v1.quant.config import KVQConfig, QADQuantConfig, SelectiveQuantConfig
from xtuner.v1.quant.export import METADATA_FILENAME, ROUNDING_SEMVER, WEIGHTS_FILENAME, export_qad, load_qad_export
from xtuner.v1.quant.fake_quant_linear import FakeQuantLinear, apply_quantization
from xtuner.v1.quant.kv_cache import KVQuantHandle, apply_kv_quantization


HIDDEN = 32
KV_HEADS = 2
HEAD_DIM = 8
NUM_BLOCKS = 3


class _DeltaOne:
    """Trivial LoRA adapter: delta_w = ones(out, in) * scale."""

    def __init__(self, shape: tuple[int, int], scale: float = 0.01) -> None:
        self.delta = torch.ones(shape) * scale

    def delta_w(self) -> torch.Tensor:
        return self.delta


class _TinyAttention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.num_key_value_heads = KV_HEADS
        self.k_proj = nn.Linear(hidden, KV_HEADS * HEAD_DIM)
        self.v_proj = nn.Linear(hidden, KV_HEADS * HEAD_DIM)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_proj(x), self.v_proj(x)


class _Block(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = _TinyAttention(hidden)
        self.mlp = nn.ModuleDict({"gate_proj": nn.Linear(hidden, hidden), "down_proj": nn.Linear(hidden, hidden)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k, v = self.self_attn(x)
        return self.mlp["down_proj"](
            self.mlp["gate_proj"](x) + k.mean(dim=-1, keepdim=True) + v.mean(dim=-1, keepdim=True)
        )


class _Inner(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(hidden) for _ in range(NUM_BLOCKS)])


class _TinyModel(nn.Module):
    def __init__(self, hidden: int = HIDDEN, vocab: int = 48) -> None:
        super().__init__()
        self.model = _Inner(hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


def _quant_cfg() -> QADQuantConfig:
    return QADQuantConfig(
        quant_schema="w8a8",
        selective=SelectiveQuantConfig(exclude=["lm_head", "*k_proj", "*v_proj"]),
        kv_cache=KVQConfig(),
    )


def _build(seed: int) -> tuple[_TinyModel, torch.Tensor]:
    torch.manual_seed(seed)
    model = _TinyModel()
    x = torch.randn(4, HIDDEN)
    return model, x


def _freeze(model: nn.Module, x: torch.Tensor) -> None:
    """Calibrate via one forward, then freeze every fake-quant module and KV
    quantizer."""
    model(x)
    for module in model.modules():
        if isinstance(module, FakeQuantLinear):
            module.freeze_scales()
        handle = getattr(module, "_qad_kv_handle", None)
        if isinstance(handle, KVQuantHandle):
            handle.quantizer.set_frozen(True)


def _frozen_scale(model: nn.Module, name: str) -> list[float]:
    module = model.get_submodule(name)
    assert isinstance(module, FakeQuantLinear) and module._w_scale_cache is not None
    return module._w_scale_cache.detach().float().reshape(-1).tolist()


class TestExport:
    def test_export_raises_on_unfrozen(self, tmp_path: Path) -> None:
        model, x = _build(0)
        apply_quantization(model, _quant_cfg())
        apply_kv_quantization(model, KVQConfig())
        with pytest.raises(RuntimeError, match="freeze"):
            export_qad(model, _quant_cfg(), tmp_path)

    def test_metadata_round_trip_and_bit_exact_replay(self, tmp_path: Path) -> None:
        cfg = _quant_cfg()
        model, x = _build(1)
        apply_quantization(model, cfg)
        apply_kv_quantization(model, cfg.kv_cache)
        _freeze(model, x)
        ref = model(x)

        report = export_qad(model, cfg, tmp_path / "export")
        assert report["num_modules"] > 0 and report["num_kv_handles"] == NUM_BLOCKS
        assert (tmp_path / "export" / WEIGHTS_FILENAME).exists()
        assert (tmp_path / "export" / METADATA_FILENAME).exists()
        assert "holdout_metrics" not in [p.name for p in (tmp_path / "export").iterdir()]

        metadata = json.loads((tmp_path / "export" / METADATA_FILENAME).read_text())
        assert set(metadata.keys()) == {"apiversion", "spec"}
        spec = metadata["spec"]
        assert spec["created_from"] == cfg.model_dump()
        assert spec["model_vocab_size"] == 48
        assert spec["rounding_semver"] == ROUNDING_SEMVER
        gate_proj = spec["modules"]["model.layers.0.mlp.gate_proj"]
        assert gate_proj["dtype"] == "int8" and gate_proj["scope"] == "per_channel"
        assert gate_proj["frozen"] is True and gate_proj["rounding_semver"] == ROUNDING_SEMVER
        scale_rec = gate_proj["scale"]
        assert scale_rec["shape"] == [HIDDEN]
        kv_rec = spec["kv"]["model.layers.0.self_attn"]
        assert kv_rec["k_scale"]["shape"] == [KV_HEADS * HEAD_DIM]
        assert kv_rec["v_scale"]["shape"] == [KV_HEADS * HEAD_DIM]
        assert kv_rec["k_scale"]["values"] != kv_rec["v_scale"]["values"]
        assert kv_rec["frozen"] is True

        # Scales must be fp32 exactly (round-trip through JSON preserves the fp32 bit pattern).
        for rec in spec["modules"].values():
            assert all(v == torch.tensor(v, dtype=torch.float32).item() for v in rec["scale"]["values"])

        replay_model, _ = _build(2)  # different init; weights come from the safetensors file
        weights = load_file(str(tmp_path / "export" / WEIGHTS_FILENAME))
        replay_model.load_state_dict(weights)
        load_qad_export(replay_model, tmp_path / "export" / METADATA_FILENAME)
        assert torch.equal(replay_model(x), ref)

    def test_scale_invariance_after_freeze(self, tmp_path: Path) -> None:
        cfg = _quant_cfg()
        model, x = _build(3)
        apply_quantization(model, cfg)
        apply_kv_quantization(model, cfg.kv_cache)
        _freeze(model, x)
        name = "model.layers.1.mlp.gate_proj"
        before = _frozen_scale(model, name)
        export_qad(model, cfg, tmp_path / "a")

        with torch.no_grad():  # weights mutate after freeze; scales must not follow
            module = model.get_submodule(name)
            assert isinstance(module, FakeQuantLinear)
            module.weight.add_(0.5)
        assert _frozen_scale(model, name) == before

        export_qad(model, cfg, tmp_path / "b")
        meta_a = json.loads((tmp_path / "a" / METADATA_FILENAME).read_text())
        meta_b = json.loads((tmp_path / "b" / METADATA_FILENAME).read_text())
        assert meta_a["spec"]["modules"][name]["scale"]["values"] == meta_b["spec"]["modules"][name]["scale"]["values"]

    def test_lora_merged_weights_exported(self, tmp_path: Path) -> None:
        cfg = _quant_cfg()
        model, x = _build(4)
        apply_quantization(model, cfg)
        apply_kv_quantization(model, cfg.kv_cache)
        name = "model.layers.2.mlp.gate_proj"
        module = model.get_submodule(name)
        assert isinstance(module, FakeQuantLinear)
        adapter = _DeltaOne((module.out_features, module.in_features), scale=0.01)
        module.attach_lora(adapter)  # type: ignore[arg-type]
        _freeze(model, x)

        export_qad(model, cfg, tmp_path)
        weights = load_file(str(tmp_path / WEIGHTS_FILENAME))
        merged = module.weight.detach() + adapter.delta_w()
        assert torch.equal(weights[f"{name}.weight"], merged.to(module.weight.dtype))

        metadata = json.loads((tmp_path / METADATA_FILENAME).read_text())
        assert metadata["spec"]["modules"][name]["lora_merged"] is True
        unmerged = metadata["spec"]["modules"]["model.layers.0.mlp.gate_proj"]
        assert unmerged["lora_merged"] is False

    def test_replayed_kv_scale_is_injected(self, tmp_path: Path) -> None:
        cfg = _quant_cfg()
        model, x = _build(5)
        apply_quantization(model, cfg)
        apply_kv_quantization(model, cfg.kv_cache)
        _freeze(model, x)
        export_qad(model, cfg, tmp_path)
        metadata = json.loads((tmp_path / METADATA_FILENAME).read_text())
        exported_k = metadata["spec"]["kv"]["model.layers.0.self_attn"]["k_scale"]["values"]

        replay_model, _ = _build(6)
        weights = load_file(str(tmp_path / WEIGHTS_FILENAME))
        replay_model.load_state_dict(weights)
        load_qad_export(replay_model, tmp_path / METADATA_FILENAME)

        handle = getattr(replay_model.get_submodule("model.layers.0.self_attn"), "_qad_kv_handle", None)
        assert isinstance(handle, KVQuantHandle)
        assert handle.quantizer.k_scale is not None
        assert handle.quantizer.k_scale.detach().float().reshape(-1).tolist() == exported_k

        # Mutating the calibration stream after replay must not move the frozen injected scale.
        big = x * 1000.0
        replay_model(big)
        assert handle.quantizer.k_scale.detach().float().reshape(-1).tolist() == exported_k
