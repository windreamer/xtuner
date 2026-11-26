"""QAD export: final master weights + quant metadata, and the replay loader
(design doc §9.1).

``export_qad`` writes an HF-compatible ``model.safetensors`` (LoRA adapters merged into the master weights) and a
``qad_quant_metadata.json`` whose top level mirrors the msmodelslim ``BaseQuantConfig{apiversion, spec}`` convention so
``modelslim_convert`` can consume it. ``load_qad_export`` replays the metadata on a fresh model: it re-applies fake
quantization and KV simulation, then injects the exported scales so the replayed forward matches the exporting run bit-
exactly (the weights file itself is loaded by the caller).
"""

import json
from pathlib import Path
from typing import Any, TypedDict

import torch
from safetensors.torch import save_file
from torch import nn

from xtuner.v1.quant.config import QADQuantConfig
from xtuner.v1.quant.fake_quant_linear import FakeQuantLinear, apply_quantization
from xtuner.v1.quant.kv_cache import KVQuantHandle, apply_kv_quantization


__all__ = ["ExportReport", "ROUNDING_SEMVER", "WEIGHTS_FILENAME", "METADATA_FILENAME", "export_qad", "load_qad_export"]


ROUNDING_SEMVER: str = "1.0.0"
"""Rounding semantic version: roundTiesToEven + clamp ``[-2^(b-1), 2^(b-1)-1]``
(design doc §4.3)."""

WEIGHTS_FILENAME: str = "model.safetensors"
METADATA_FILENAME: str = "qad_quant_metadata.json"
_APIVERSION: str = "modelslim_v1"
_KV_HANDLE_ATTR: str = "_qad_kv_handle"
_LEVEL: int = 127  # int8 symmetric level: scale = amax / 127 (§6.3)


class ExportReport(TypedDict):
    """Paths and summary of a successful :func:`export_qad` call.

    Args:
        weights_path (str): Path of the written ``model.safetensors``.
        metadata_path (str): Path of the written ``qad_quant_metadata.json``.
        num_modules (int): Number of exported ``FakeQuantLinear`` module records.
        num_kv_handles (int): Number of exported KV quantizer records.
        model_vocab_size (int): Vocabulary size recorded in the metadata.
    """

    weights_path: str
    metadata_path: str
    num_modules: int
    num_kv_handles: int
    model_vocab_size: int


def _scale_record(scale: torch.Tensor) -> dict[str, Any]:
    """Serialize an fp32 scale tensor as ``{shape, values}`` for JSON
    embedding."""
    s = scale.detach().float().cpu().reshape(-1)
    return {"shape": list(scale.shape), "values": s.tolist()}


def _model_vocab_size(model: nn.Module) -> int:
    """Infer the model vocabulary size from ``lm_head``/embedding modules or a
    ``vocab_size`` attribute."""
    for name, module in model.named_modules():
        if name.endswith("lm_head") and isinstance(module, nn.Linear):
            return int(module.out_features)
    vocab_size = getattr(model, "vocab_size", None)
    if isinstance(vocab_size, int):
        return vocab_size
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return int(module.num_embeddings)
    raise ValueError("export_qad: cannot infer model_vocab_size (no lm_head, vocab_size attr, or embedding found)")


def _kv_handles(model: nn.Module) -> dict[str, KVQuantHandle]:
    """Collect the attached KV handles keyed by their attention module name."""
    handles: dict[str, KVQuantHandle] = {}
    for name, module in model.named_modules():
        handle = getattr(module, _KV_HANDLE_ATTR, None)
        if isinstance(handle, KVQuantHandle):
            handles[name] = handle
    return handles


def _merged_weight(linear: FakeQuantLinear) -> torch.Tensor:
    """Return ``W_eff = W + sum(delta_w)`` in the master weight's dtype."""
    w = linear.weight.detach()
    dtype = w.dtype
    w_merged = w.float()
    for adapter in linear._lora_adapters:
        w_merged = w_merged + adapter.delta_w().to(torch.float32)
    return w_merged.to(dtype)


def _module_records(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Build the per-module metadata records; raises if any module is
    unfrozen."""
    records: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, FakeQuantLinear):
            continue
        if not module._scales_frozen:
            raise RuntimeError(
                f"export_qad: module '{name}' is not scale-frozen; export must run in or after the freeze phase "
                "(design doc §5.3/§9.2: the deployment scale must equal the training-end scale)"
            )
        w_eff = module._effective_weight()
        if module._w_scale_cache is None:
            # Frozen but never forwarded after freezing: the next forward would compute exactly this scale, so
            # computing it here and pinning it is bit-identical to the training-end semantics.
            module._w_scale_cache = module._weight_quantizer.compute_scale(w_eff)
        scale = module._w_scale_cache
        assert scale is not None
        records[name] = {
            "dtype": module._weight_quantizer.dtype,
            "scope": module._weight_quantizer.scope,
            "group_size": getattr(module._weight_quantizer, "group_size", None),
            "scale": _scale_record(scale),
            "frozen": True,
            "lora_merged": len(module._lora_adapters) > 0,
            "rounding_semver": ROUNDING_SEMVER,
        }
    return records


def _kv_records(handles: dict[str, KVQuantHandle]) -> dict[str, dict[str, Any]]:
    """Build the per-attention KV metadata records; raises if any handle is
    unfrozen."""
    records: dict[str, dict[str, Any]] = {}
    for name, handle in handles.items():
        quantizer = handle.quantizer
        if not quantizer.frozen:
            raise RuntimeError(
                f"export_qad: KV quantizer on '{name}' is not frozen; export must run in or after the freeze phase "
                "(design doc §5.3/§9.2)"
            )
        k_scale = quantizer.k_scale
        v_scale = quantizer.v_scale
        if k_scale is None or v_scale is None:
            raise RuntimeError(
                f"export_qad: KV quantizer on '{name}' has no calibrated scale (no tensor seen before freezing); "
                "run at least one forward before freezing"
            )
        records[name] = {
            "k_scale": _scale_record(k_scale),
            "v_scale": _scale_record(v_scale),
            "ema_decay": quantizer.ema_decay,
            "frozen": True,
            "rounding_semver": ROUNDING_SEMVER,
        }
    return records


def export_qad(model: nn.Module, quant_cfg: QADQuantConfig, output_dir: Path) -> ExportReport:
    """Export the final master weights and quant metadata of a QAD-trained
    model (design doc §9.1).

    Writes ``model.safetensors`` — the master weights with any LoRA adapters merged into
    them (``W_eff = W + delta_w``, merge-then-quant §5.2), HF-compatible layout with
    unchanged keys — and ``qad_quant_metadata.json`` with the ``{apiversion, spec}`` top
    level of the msmodelslim ``BaseQuantConfig`` convention. ``spec`` carries one record
    per ``FakeQuantLinear`` (schema/granularity/group_size/dtype/scope/fp32 scale/frozen
    flag/``lora_merged`` flag/``rounding_semver``) and one record per attached
    :class:`KVQuantHandle` (independent fp32 ``k_scale``/``v_scale`` with shapes and
    values, EMA decay, frozen flag), plus ``model_vocab_size`` and ``created_from`` (the
    echo of ``quant_cfg.model_dump()``).

    The export MUST be called in or after the scale freeze phase (design doc §5.3): it
    raises ``RuntimeError`` if any ``FakeQuantLinear`` or KV quantizer is unfrozen, since
    deployment scale == training-end scale is the premise of the §9.2 KL chain.

    Args:
        model (nn.Module): Quantized model in the frozen state; mutated only in that a
            frozen-but-uncached module scale is pinned from its current ``W_eff``.
        quant_cfg (QADQuantConfig): Quantization config echoed into the metadata.
        output_dir (Path): Directory to create; overwritten files inside it.

    Returns:
        ExportReport: Written artifact paths and record counts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module_records = _module_records(model)
    kv_records = _kv_records(_kv_handles(model))
    vocab_size = _model_vocab_size(model)

    metadata: dict[str, Any] = {
        "apiversion": _APIVERSION,
        "spec": {
            "modules": module_records,
            "kv": kv_records,
            "rounding_semver": ROUNDING_SEMVER,
            "model_vocab_size": vocab_size,
            "created_from": quant_cfg.model_dump(),
        },
    }

    state_dict = model.state_dict()
    weights: dict[str, torch.Tensor] = {}
    lora_names = {name for name, rec in module_records.items() if rec["lora_merged"]}
    for key, tensor in state_dict.items():
        merged = False
        for module_name in lora_names:
            if key == f"{module_name}.weight":
                module = model.get_submodule(module_name)
                assert isinstance(module, FakeQuantLinear)
                weights[key] = _merged_weight(module).contiguous()
                merged = True
                break
        if not merged:
            weights[key] = tensor.detach().to(tensor.dtype).cpu().contiguous()
    weights_path = output_dir / WEIGHTS_FILENAME
    save_file(weights, str(weights_path), metadata={"format": "pt"})

    metadata_path = output_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ExportReport(
        weights_path=str(weights_path),
        metadata_path=str(metadata_path),
        num_modules=len(module_records),
        num_kv_handles=len(kv_records),
        model_vocab_size=vocab_size,
    )


def _inject_module_scales(model: nn.Module, records: dict[str, dict[str, Any]]) -> None:
    """Pin the exported weight scales onto the freshly quantized model."""
    for name, rec in records.items():
        module = model.get_submodule(name)
        if not isinstance(module, FakeQuantLinear):
            raise RuntimeError(f"load_qad_export: module '{name}' is not a FakeQuantLinear after apply_quantization")
        scale = torch.tensor(rec["scale"]["values"], dtype=torch.float32).reshape(rec["scale"]["shape"])
        module._w_scale_cache = scale
        module._scales_frozen = True


def _inject_kv_scales(model: nn.Module, records: dict[str, dict[str, Any]]) -> None:
    """Pin the exported K/V scales onto the freshly attached KV quantizers."""
    for name, rec in records.items():
        module = model.get_submodule(name)
        handle = getattr(module, _KV_HANDLE_ATTR, None)
        if not isinstance(handle, KVQuantHandle):
            raise RuntimeError(f"load_qad_export: no KVQuantHandle attached at '{name}'")
        quantizer = handle.quantizer
        quantizer.k_amax_ema = (
            torch.tensor(rec["k_scale"]["values"], dtype=torch.float32).reshape(rec["k_scale"]["shape"]) * _LEVEL
        )
        quantizer.v_amax_ema = (
            torch.tensor(rec["v_scale"]["values"], dtype=torch.float32).reshape(rec["v_scale"]["shape"]) * _LEVEL
        )
        quantizer.set_frozen(True)


def load_qad_export(model: nn.Module, metadata_path: Path) -> None:
    """Replay a QAD export onto a fresh model (design doc §9.1, §9.2 gate C).

    Applies :func:`apply_quantization` and, when configured,
    :func:`apply_kv_quantization` using the ``created_from`` config echo, then injects
    the exported fp32 scales in place of freshly computed amax, freezing them. With the
    exported weights file loaded into ``model`` by the caller, the replayed forward
    matches the exporting run bit-exactly (same weights, same scales, same §4.3 formulas).

    Args:
        model (nn.Module): Fresh (unquantized) model whose weights are already loaded from
            ``model.safetensors``; quantized in place.
        metadata_path (Path): Path of the ``qad_quant_metadata.json`` written by
            :func:`export_qad`.
    """
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    spec = metadata["spec"]
    quant_cfg = QADQuantConfig.model_validate(spec["created_from"])
    apply_quantization(model, quant_cfg)
    if quant_cfg.kv_cache is not None:
        apply_kv_quantization(model, quant_cfg.kv_cache, quant_cfg.kv_amax_ema_decay)
    _inject_module_scales(model, spec["modules"])
    if spec["kv"]:
        _inject_kv_scales(model, spec["kv"])
