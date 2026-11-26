"""FakeQuantLinear, selective quantization apply/remove, and LoRA composition
(design doc §5).

The quantized layer is implemented as an in-place ``__class__`` swap of an existing ``nn.Linear`` instance: parameter
names (``weight``/``bias``) and therefore the ``state_dict`` keys are preserved exactly, and FSDP wrapping sees a plain
``nn.Linear``.
"""

import fnmatch
import re
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

from xtuner.v1.quant.config import QADQuantConfig, SelectiveQuantConfig
from xtuner.v1.quant.quantizer import IntQuantizer, Quantizer


# Decoder-block index extraction pattern: matches module names like
# "model.layers.3.mlp.gate_proj" and captures the block index (here ``3``).
_BLOCK_INDEX_PATTERN = re.compile(r"layers\.(\d+)\.")

__all__ = ["FakeQuantLinear", "LoRAAdapter", "apply_quantization", "build_quantizers", "remove_quantization"]


class LoRAAdapter(Protocol):
    """Minimal adapter protocol for merge-then-quant LoRA composition (design
    doc §5.2).

    Implementations expose ``delta_w()`` returning the effective low-rank update ``B @ A`` as a tensor broadcastable to
    the wrapped ``nn.Linear`` weight shape ``(out_features, in_features)``. A full LoRA trainer is out of scope here
    (T7 owns the training wiring).
    """

    def delta_w(self) -> torch.Tensor:
        """Return the low-rank weight update broadcastable to the weight shape.

        Returns:
            torch.Tensor: Update tensor added to ``weight`` before fake quantization.
        """
        ...


def build_quantizers(cfg: QADQuantConfig) -> tuple[Quantizer, Quantizer]:
    """Build the (weight, activation) quantizer pair for a quantization config.

    Args:
        cfg (QADQuantConfig): Quantization schema config.

    Returns:
        tuple[Quantizer, Quantizer]: Weight quantizer and activation quantizer per
        design §4.2: ``w8a8`` -> int8 per-channel weight + int8 per-token activation;
        ``w4a8`` -> int4 per-group (``group_size``) or per-channel weight (per
        ``w4_granularity``) + int8 per-batch activation.
    """
    if cfg.quant_schema == "w8a8":
        weight_q: Quantizer = IntQuantizer(8, "per_channel")
        act_q: Quantizer = IntQuantizer(8, "per_token")
    else:
        if cfg.w4_granularity == "per_group":
            weight_q = IntQuantizer(4, "per_group", group_size=cfg.group_size)
        else:
            weight_q = IntQuantizer(4, "per_channel")
        act_q = IntQuantizer(8, "per_batch")
    return weight_q, act_q


class FakeQuantLinear(nn.Linear):
    """``nn.Linear`` with weight/activation fake quantization (design doc
    §5.2).

    Instances are created either directly or by converting an existing ``nn.Linear``
    in place via :func:`apply_quantization`; in both cases the ``state_dict`` keys are
    exactly those of a plain ``nn.Linear``. Weight scales are computed once per
    training step (:meth:`begin_step` invalidates the cache) and reused across
    micro-batches; activation scales are recomputed on every forward.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether the layer has a bias term.
        quant_cfg (QADQuantConfig): Quantization schema config; quantizers are built
            from it internally.
    """

    _weight_quantizer: Quantizer
    _act_quantizer: Quantizer
    _lora_adapters: list[LoRAAdapter]
    _w_scale_cache: torch.Tensor | None
    _scales_frozen: bool

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, quant_cfg: QADQuantConfig | None = None
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        if quant_cfg is None:
            quant_cfg = QADQuantConfig()
        self._init_quant_state(quant_cfg)

    @classmethod
    def _convert(cls, linear: nn.Linear, quant_cfg: QADQuantConfig) -> "FakeQuantLinear":
        """Convert an ``nn.Linear`` instance into a ``FakeQuantLinear`` in
        place.

        Args:
            linear (nn.Linear): The linear layer to convert; mutated in place.
            quant_cfg (QADQuantConfig): Quantization schema config.

        Returns:
            FakeQuantLinear: The same object, re-typed.
        """
        linear.__class__ = cls
        assert isinstance(linear, FakeQuantLinear)
        linear._init_quant_state(quant_cfg)
        return linear

    def _init_quant_state(self, quant_cfg: QADQuantConfig) -> None:
        self._weight_quantizer, self._act_quantizer = build_quantizers(quant_cfg)
        self._lora_adapters = []
        self._w_scale_cache = None
        self._scales_frozen = False

    def begin_step(self) -> None:
        """Invalidate the cached weight scale for a new optimizer step (design
        doc §5.3).

        Called by the trainer once per optimizer step; within a step repeated forward calls (micro-batches / grad
        accumulation) reuse the cached scale. No-op while scales are frozen.
        """
        if not self._scales_frozen:
            self._w_scale_cache = None

    def freeze_scales(self) -> None:
        """Freeze the weight scale: it stays fixed while weights keep training
        (design doc §5.3).

        The scale is snapshotted from the current cache, or computed from the current ``W_eff`` on the next forward if
        none exists, and then pinned: subsequent forwards re-merge LoRA deltas and fake-quantize the *current* weight
        with this fixed scale, so gradients keep flowing to the master weights via STE. This matches §5.3
        ("停止一切重算，scale 定格") and the §9.2 KL-chain requirement that export quantizes the final weights with the same
        pinned scale.
        """
        self._scales_frozen = True

    def unfreeze_scales(self) -> None:
        """Unfreeze scales; the cached scale is cleared so the next forward
        recomputes it."""
        self._scales_frozen = False
        self._w_scale_cache = None

    def attach_lora(self, adapter: LoRAAdapter) -> None:
        """Attach a LoRA adapter whose delta is merged into the weight before
        quantization.

        Args:
            adapter (LoRAAdapter): Adapter exposing ``delta_w()``; multiple adapters sum.
        """
        self._lora_adapters.append(adapter)

    def detach_lora(self, adapter: LoRAAdapter | None = None) -> None:
        """Detach a LoRA adapter, or all of them when ``adapter`` is ``None``.

        Args:
            adapter (LoRAAdapter | None): Specific adapter to remove; ``None`` removes all.
        """
        if adapter is None:
            self._lora_adapters.clear()
        elif adapter in self._lora_adapters:
            self._lora_adapters.remove(adapter)

    def _effective_weight(self) -> torch.Tensor:
        w = self.weight
        for adapter in self._lora_adapters:
            w = w + adapter.delta_w()
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantized forward: merge LoRA, quantize weight (cached scale),
        quantize input.

        Args:
            x (torch.Tensor): Input activations; the last dim must equal ``in_features``.

        Returns:
            torch.Tensor: ``F.linear(quant(x), quant(W_eff), bias)`` with dequantized
            tensors in their input dtypes.
        """
        w_eff = self._effective_weight()
        if self._w_scale_cache is None:
            self._w_scale_cache = self._weight_quantizer.compute_scale(w_eff)
        w_q = self._weight_quantizer.quant_dequant(w_eff, self._w_scale_cache)
        x_q = self._act_quantizer.quant_dequant(x, self._act_quantizer.compute_scale(x))
        return F.linear(x_q, w_q, self.bias)


def _block_index(name: str) -> int | None:
    match = _BLOCK_INDEX_PATTERN.search(name)
    return int(match.group(1)) if match is not None else None


def _num_blocks(names: list[str]) -> int:
    indices = [idx for name in names if (idx := _block_index(name)) is not None]
    return max(indices) + 1 if indices else 0


def _is_selected(name: str, num_blocks: int, selective: SelectiveQuantConfig) -> bool:
    if any(fnmatch.fnmatch(name, pat) for pat in selective.exclude):
        return False
    if not any(fnmatch.fnmatch(name, pat) for pat in selective.include):
        return False
    idx = _block_index(name)
    if idx is not None:
        if idx < selective.exclude_first_k_blocks:
            return False
        if idx >= num_blocks - selective.exclude_last_k_blocks:
            return False
    return True


def apply_quantization(model: nn.Module, cfg: QADQuantConfig) -> None:
    """Convert selected ``nn.Linear`` modules of ``model`` into
    ``FakeQuantLinear`` in place.

    Module selection follows ``cfg.selective`` (design doc §4.1): ``include`` fnmatch
    patterns are matched against full module names (default ``["*"]``), ``exclude``
    patterns take precedence, and ``exclude_first_k_blocks``/``exclude_last_k_blocks``
    exclude decoder blocks by index extracted with the pattern ``r"layers\\.(\\d+)\\."``
    (block count inferred as max index + 1). Only ``nn.Linear`` instances are touched;
    embeddings, ``lm_head`` and norms stay bf16 unless a recipe's exclude/include
    patterns say otherwise. ``state_dict`` keys are unchanged, so checkpoints stay
    compatible in both directions. Re-applying to a model that already contains
    quantized modules raises ``RuntimeError``; call :func:`remove_quantization` first.

    Args:
        model (nn.Module): Model to quantize in place; apply before FSDP wrapping.
        cfg (QADQuantConfig): Quantization schema and selective policy.
    """
    names = [name for name, _ in model.named_modules()]
    num_blocks = _num_blocks(names)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, FakeQuantLinear):
            raise RuntimeError(f"apply_quantization: module '{name}' is already quantized; remove_quantization first")
        if not _is_selected(name, num_blocks, cfg.selective):
            continue
        FakeQuantLinear._convert(module, cfg)


def _detach_attr(module: nn.Module, name: str) -> None:
    if name in module._modules:
        del module._modules[name]
    module.__dict__.pop(name, None)


def remove_quantization(model: nn.Module) -> None:
    """Restore all ``FakeQuantLinear`` modules of ``model`` to plain
    ``nn.Linear`` in place.

    Quantizer, LoRA and scale-cache state is detached; parameters and their names are
    untouched, so the model behaves exactly as before :func:`apply_quantization` and
    ``state_dict`` keys are unchanged. Non-quantized modules are left alone.

    Args:
        model (nn.Module): Model to restore in place.
    """
    for module in model.modules():
        if isinstance(module, FakeQuantLinear):
            module.__class__ = nn.Linear  # type: ignore[assignment]
            _detach_attr(module, "_weight_quantizer")
            _detach_attr(module, "_act_quantizer")
            _detach_attr(module, "_lora_adapters")
            _detach_attr(module, "_w_scale_cache")
            _detach_attr(module, "_scales_frozen")
