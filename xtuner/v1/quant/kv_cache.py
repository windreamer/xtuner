"""KV-cache quantization simulation for QAD training (design doc §6, §4.3.7).

Deploy-side inference quantizes K/V symmetrically to int8 when they are written into the KV cache
and dequantizes on read. Training simulates this with a quantize-dequantize applied to K and V
before they enter the cache or the attention computation (§6.1); Q, O and the softmax internals
carry no KV quantization, and gradients flow back to ``k_proj``/``v_proj`` through the STE.

Numerics (§4.3 canonical formula, per-(KV-head, head-dim) channels): input ``K`` of shape
``(B, H_kv, S, D)`` is transposed to ``(B, S, H_kv, D)``, reshaped to ``(B*S, H_kv*D)``, and its
scale is reduced over ``dim=0`` (batch·seq), giving scale shape ``(H_kv*D,)`` that broadcasts back
over the same view. K and V keep independent scale sets. ``scale = max(amax/127, eps_fp32)`` with
zero-containment, ``q = clamp(round_half_even(x/scale), -128, 127)``, ``x_hat = q * scale``.

Scale source (§6.3): per-step amax with EMA smoothing ``ema = decay*ema + (1-decay)*amax_new``
(init ``ema = first amax``), frozen scales are kept fixed no matter which tensors pass through
(§5.3 freeze phase). The EMA state lives in non-persistent buffers, so it never leaks into
``state_dict``.

Attach mechanism (§6.2). :func:`apply_kv_quantization` selects attention modules by a documented
pattern (class name contains ``"Attention"`` or ``"MLA"``, or the module-name path contains
``"self_attn"``/``"attention"``; bare ``nn.Linear`` are never selected, and neither are
quantizer-owned submodules — ``Quantizer`` instances such as the ``IntQuantizer``
``_weight_quantizer``/``_act_quantizer`` children that ``apply_quantization`` registers on
``FakeQuantLinear``, or any module under a path segment ending in ``_quantizer``) and wires
quantization at the module's K/V entry points:

1. Forward-argument interception — if the module's ``forward`` signature has parameters named
   ``k``/``v``/``key``/``value``/``keys``/``values``, the bound method is patched so those tensor
   arguments are quantize-dequantized before the original forward runs.
2. Projection-boundary interception — xtuner attention modules (``MultiHeadAttention``,
   ``MultiLatentAttention``) compute K/V internally, so their ``k_proj``/``v_proj`` outputs are
   quantized (K/V exactly where they enter cache/attention); the fused MLA ``kv_b_proj`` output is
   split into its ``qk_nope_head_dim`` (K) and ``v_head_dim`` (V) halves and quantized with
   independent scale sets.

Both interceptions share one :class:`KVCacheQuantizer` per attention module. Removal is symmetric:
``handle.detach()`` or :func:`remove_kv_quantization` restores the original forwards bit-exactly.
Attachment is idempotent: re-applying to an already-wrapped module returns the existing handle
instead of stacking a second quantization.
"""

import functools
import inspect
from typing import Any, Literal

import torch
from torch import nn

from .config import KVQConfig
from .quantizer import Quantizer, _QuantDequantSTE


_EPS_FP32: float = torch.finfo(torch.float32).eps
_LEVEL: int = 127  # int8 symmetric levels: [-128, 127], scale = amax / 127 (§6.3)
_QMIN: int = -(2**7)
_QMAX: int = 2**7 - 1

_KV_ARG_ROLES: dict[str, Literal["key", "value"]] = {
    "k": "key",
    "key": "key",
    "keys": "key",
    "v": "value",
    "value": "value",
    "values": "value",
}
_HANDLE_ATTR: str = "_qad_kv_handle"
_QUANTIZER_ATTR: str = "_qad_kv_quantizer"


class _Missing:
    """Sentinel marking "attribute was absent before patching" in restore
    records."""


_MISSING: _Missing = _Missing()


def _channel_amax(x2d: torch.Tensor) -> torch.Tensor:
    """Zero-contained symmetric amax per column of ``(B*S, H_kv*D)`` statistics
    (§4.3)."""
    x = x2d.detach().float()
    return torch.maximum(x.min(dim=0).values.clamp(max=0).neg(), x.max(dim=0).values.clamp(min=0))


class KVCacheQuantizer(nn.Module):
    """Per-(KV-head, head-dim) symmetric int8 KV fake-quantizer with EMA scale
    tracking (§6).

    K and V hold independent EMA states and therefore independent scale sets. All state lives in
    non-persistent buffers, so nothing leaks into ``state_dict``.

    Args:
        ema_decay (float): EMA decay ``kv_amax_ema_decay`` for amax smoothing, in ``[0, 1)``;
            the effective amax updates as ``ema = decay*ema + (1-decay)*amax_new`` and is
            initialized with the first observed amax (§6.3).
    """

    def __init__(self, ema_decay: float = 0.999) -> None:
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"KVCacheQuantizer: ema_decay must be in [0, 1), got {ema_decay}")
        super().__init__()
        self.ema_decay: float = float(ema_decay)
        self._frozen: bool = False
        self.k_amax_ema: torch.Tensor | None
        self.v_amax_ema: torch.Tensor | None
        self.register_buffer("k_amax_ema", None, persistent=False)
        self.register_buffer("v_amax_ema", None, persistent=False)

    @property
    def frozen(self) -> bool:
        """Bool: Whether scales are frozen (§5.3 freeze phase)."""
        return self._frozen

    @property
    def k_scale(self) -> torch.Tensor | None:
        """torch.Tensor | None: Current fp32 key scale of shape ``(H_kv*D,)``,
        or ``None`` before the first calibration tensor."""
        ema = self.k_amax_ema
        return None if ema is None else torch.clamp(ema.float() / _LEVEL, min=_EPS_FP32)

    @property
    def v_scale(self) -> torch.Tensor | None:
        """torch.Tensor | None: Current fp32 value scale of shape
        ``(H_kv*D,)``, or ``None`` before the first calibration tensor."""
        ema = self.v_amax_ema
        return None if ema is None else torch.clamp(ema.float() / _LEVEL, min=_EPS_FP32)

    def set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the scales (§5.3).

        While frozen, neither the EMA nor the effective scale updates regardless of the tensors
        passing through; unfreezing resumes EMA updates.

        Args:
            frozen (bool): ``True`` freezes both K and V scales, ``False`` resumes updates.
        """
        self._frozen = bool(frozen)

    def quantize_key(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize a key tensor per (head, head_dim) channel.

        Args:
            x (torch.Tensor): Key tensor of shape ``(B, H_kv, S, D)``.

        Returns:
            torch.Tensor: Quantize-dequantized key with the same shape and dtype as ``x``; the
            scale statistics and EMA are this quantizer's key state.
        """
        return self._quantize(x, is_key=True)

    def quantize_value(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize a value tensor per (head, head_dim) channel.

        Args:
            x (torch.Tensor): Value tensor of shape ``(B, H_kv, S, D)``.

        Returns:
            torch.Tensor: Quantize-dequantized value with the same shape and dtype as ``x``; the
            scale statistics and EMA are this quantizer's independent value state.
        """
        return self._quantize(x, is_key=False)

    def _quantize(self, x: torch.Tensor, is_key: bool) -> torch.Tensor:
        if x.numel() == 0:
            raise ValueError("KVCacheQuantizer: cannot quantize an empty tensor")
        if x.dim() != 4:
            raise ValueError(f"KVCacheQuantizer: expected a 4-D (B, H_kv, S, D) tensor, got shape {tuple(x.shape)}")
        b, h, s, d = x.shape
        # Axis contract (§4.3.7): (B, H_kv, S, D) -> (B, S, H_kv, D) -> (B*S, H_kv*D), reduce dim=0.
        x2d = x.permute(0, 2, 1, 3).reshape(b * s, h * d)
        ema = self.k_amax_ema if is_key else self.v_amax_ema
        if self._frozen:
            if ema is None:
                raise RuntimeError("KVCacheQuantizer: frozen before any calibration tensor was seen; unfreeze first")
        else:
            amax = _channel_amax(x2d)
            ema = amax if ema is None else self.ema_decay * ema + (1.0 - self.ema_decay) * amax
            if is_key:
                self.k_amax_ema = ema
            else:
                self.v_amax_ema = ema
        scale = torch.clamp(ema.float() / _LEVEL, min=_EPS_FP32)
        out2d = _QuantDequantSTE.apply(x2d, scale.unsqueeze(0), _QMIN, _QMAX)
        return out2d.reshape(b, s, h, d).permute(0, 2, 1, 3)


class KVQuantHandle:
    """Detachable binding between one attention module and its
    :class:`KVCacheQuantizer`.

    Attributes:
        module (nn.Module): The wrapped attention module.
        quantizer (KVCacheQuantizer): The quantizer driving the module's K/V fake-quantization.
    """

    def __init__(self, module: nn.Module, quantizer: KVCacheQuantizer) -> None:
        self.module: nn.Module = module
        self.quantizer: KVCacheQuantizer = quantizer
        self._restores: list[tuple[nn.Module, str, object]] = []
        self._detached: bool = False

    def detach(self) -> None:
        """Restore the wrapped module to its original forward behavior
        (idempotent).

        Removes every patched ``forward`` and the handle/quantizer attributes, so outputs are bit-identical to the pre-
        attach module.
        """
        if self._detached:
            return
        for target, name, original in self._restores:
            if isinstance(original, _Missing):
                delattr(target, name)
            else:
                setattr(target, name, original)
        self._restores = []
        if getattr(self.module, _QUANTIZER_ATTR, None) is self.quantizer:
            delattr(self.module, _QUANTIZER_ATTR)
        delattr(self.module, _HANDLE_ATTR)
        self._detached = True

    def _add_restore(self, target: nn.Module, name: str, original: object) -> None:
        self._restores.append((target, name, original))


def _quantize_flat_sequence(
    x: torch.Tensor, quantizer: KVCacheQuantizer, role: Literal["key", "value"], heads: int, width: int
) -> torch.Tensor:
    """Quantize a projection output of shape ``(B, S, heads*width)`` (or
    ``(B*S, heads*width)``).

    Args:
        x (torch.Tensor): Projection output whose trailing dim is ``heads*width``.
        quantizer (KVCacheQuantizer): Quantizer providing the per-(head, head_dim) scales.
        role (Literal["key", "value"]): Which independent scale set to use.
        heads (int): Number of KV heads.
        width (int): Per-head width ``D`` of the projection output.

    Returns:
        torch.Tensor: Quantize-dequantized tensor with the same shape as ``x``.
    """
    if x.dim() == 2:
        b, s = x.shape[0], 1
    elif x.dim() == 3:
        b, s = x.shape[0], x.shape[1]
    else:
        raise ValueError(
            f"apply_kv_quantization: projection output must be 2-D or 3-D (B[, S], heads*width), got shape "
            f"{tuple(x.shape)}"
        )
    x4 = x.reshape(b, s, heads, width).transpose(1, 2)
    out4 = quantizer.quantize_key(x4) if role == "key" else quantizer.quantize_value(x4)
    return out4.transpose(1, 2).reshape(x.shape)


def _apply_role(quantizer: KVCacheQuantizer, x: torch.Tensor, role: Literal["key", "value"]) -> torch.Tensor:
    if role == "key":
        return quantizer.quantize_key(x)
    return quantizer.quantize_value(x)


def _patch_forward_args(module: nn.Module, quantizer: KVCacheQuantizer, handle: KVQuantHandle) -> bool:
    """Patch ``module.forward`` so K/V-named tensor arguments are quantize-
    dequantized (§6.2).

    Args:
        module (nn.Module): Attention module to patch in place.
        quantizer (KVCacheQuantizer): Quantizer applied to the intercepted arguments.
        handle (KVQuantHandle): Handle recording how to undo the patch.

    Returns:
        bool: ``True`` if the signature exposes K/V-named parameters, ``False`` otherwise (the
        caller should fall back to projection-boundary interception).
    """
    try:
        params = list(inspect.signature(module.forward).parameters.values())
    except (TypeError, ValueError):
        return False
    positional: list[tuple[int, Literal["key", "value"]]] = []
    keyword: dict[str, Literal["key", "value"]] = {}
    pos_index = 0
    has_var_positional = False
    for param in params:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        role = _KV_ARG_ROLES.get(param.name)
        if role is None:
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                pos_index += 1
            continue
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional.append((pos_index, role))
            pos_index += 1
            keyword[param.name] = role
        else:  # KEYWORD_ONLY
            keyword[param.name] = role
    if not positional and not keyword:
        return False
    if has_var_positional:
        positional = []

    original = module.forward

    @functools.wraps(original)
    def forward(*args: Any, **kwargs: Any) -> Any:
        new_args = list(args)
        for idx, role in positional:
            if idx < len(new_args) and isinstance(new_args[idx], torch.Tensor):
                new_args[idx] = _apply_role(quantizer, new_args[idx], role)
        for name, role in keyword.items():
            value = kwargs.get(name)
            if isinstance(value, torch.Tensor):
                kwargs[name] = _apply_role(quantizer, value, role)
        return original(*new_args, **kwargs)

    module.forward = forward  # type: ignore[method-assign]
    handle._add_restore(module, "forward", _MISSING)
    return True


def _num_kv_heads(module: nn.Module) -> int:
    for attr in ("num_key_value_heads", "num_attention_heads"):
        value = getattr(module, attr, None)
        if isinstance(value, int):
            return value
    raise ValueError(
        f"apply_kv_quantization: attention module {type(module).__name__!r} exposes neither "
        f"'num_key_value_heads' nor 'num_attention_heads'; cannot derive the KV head count"
    )


def _patch_proj_output(
    proj: nn.Linear,
    quantizer: KVCacheQuantizer,
    role: Literal["key", "value"],
    heads: int,
    width: int,
    handle: KVQuantHandle,
) -> None:
    """Patch a ``k_proj``/``v_proj`` child so its output K/V tensor is fake-
    quantized (§6.2)."""
    original = proj.forward

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        if not isinstance(out, torch.Tensor):
            return out
        return _quantize_flat_sequence(out, quantizer, role, heads, width)

    proj.forward = wrapped  # type: ignore[method-assign]
    handle._add_restore(proj, "forward", _MISSING)


def _patch_kv_split_output(
    proj: nn.Linear, quantizer: KVCacheQuantizer, heads: int, k_width: int, v_width: int, handle: KVQuantHandle
) -> None:
    """Patch the fused MLA ``kv_b_proj`` child: split into K/V halves with
    independent scales."""
    original = proj.forward

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        if not isinstance(out, torch.Tensor):
            return out
        if out.dim() == 2:
            b, s = out.shape[0], 1
        elif out.dim() == 3:
            b, s = out.shape[0], out.shape[1]
        else:
            raise ValueError(
                f"apply_kv_quantization: fused kv_b_proj output must be 2-D or 3-D, got shape {tuple(out.shape)}"
            )
        out4 = out.reshape(b, s, heads, k_width + v_width)
        k4 = quantizer.quantize_key(out4[..., :k_width].transpose(1, 2)).transpose(1, 2)
        v4 = quantizer.quantize_value(out4[..., k_width:].transpose(1, 2)).transpose(1, 2)
        return torch.cat([k4, v4], dim=-1).reshape(out.shape)

    proj.forward = wrapped  # type: ignore[method-assign]
    handle._add_restore(proj, "forward", _MISSING)


def _patch_projections(module: nn.Module, quantizer: KVCacheQuantizer, handle: KVQuantHandle) -> None:
    """Wire quantization at the K/V-producing projections of a real xtuner
    attention module.

    Args:
        module (nn.Module): Attention module whose ``k_proj``/``v_proj``/``kv_b_proj`` children are
            patched in place.
        quantizer (KVCacheQuantizer): Quantizer applied to the projection outputs.
        handle (KVQuantHandle): Handle recording how to undo the patches.

    Raises:
        ValueError: If the module exposes none of ``k_proj``/``v_proj``/``kv_b_proj``.
    """
    children = dict(module.named_children())
    heads = _num_kv_heads(module)
    patched = False
    k_proj = children.get("k_proj")
    v_proj = children.get("v_proj")
    kv_b_proj = children.get("kv_b_proj")
    if isinstance(k_proj, nn.Linear):
        width = k_proj.out_features // heads
        _patch_proj_output(k_proj, quantizer, "key", heads, width, handle)
        patched = True
    if isinstance(v_proj, nn.Linear):
        width = v_proj.out_features // heads
        _patch_proj_output(v_proj, quantizer, "value", heads, width, handle)
        patched = True
    if isinstance(kv_b_proj, nn.Linear):
        k_width = getattr(module, "qk_nope_head_dim", None)
        v_width = getattr(module, "v_head_dim", None)
        if not isinstance(k_width, int) or not isinstance(v_width, int):
            raise ValueError(
                f"apply_kv_quantization: fused-kv module {type(module).__name__!r} lacks "
                f"'qk_nope_head_dim'/'v_head_dim'; cannot split kv_b_proj into K/V halves"
            )
        if kv_b_proj.out_features != heads * (k_width + v_width):
            raise ValueError(
                f"apply_kv_quantization: kv_b_proj.out_features={kv_b_proj.out_features} does not match "
                f"heads*(qk_nope+v_head)={heads * (k_width + v_width)} on {type(module).__name__!r}"
            )
        _patch_kv_split_output(kv_b_proj, quantizer, heads, k_width, v_width, handle)
        patched = True
    if not patched:
        raise ValueError(
            f"apply_kv_quantization: attention module {type(module).__name__!r} exposes no K/V entry point "
            f"(expected forward(k, v)-style arguments or k_proj/v_proj/kv_b_proj children)"
        )


def _is_quant_internal(name: str, module: nn.Module) -> bool:
    """Detect quantizer-owned submodules that must never be selected as
    attention roots.

    ``apply_quantization`` (``fake_quant_linear``) registers ``IntQuantizer`` submodules
    (``_weight_quantizer``/``_act_quantizer``) on ``FakeQuantLinear``, and an attached :class:`KVCacheQuantizer` itself
    becomes a ``_qad_kv_quantizer`` submodule of its attention module. Their ``named_modules()`` paths (e.g.
    ``"...self_attn.k_proj._act_quantizer"``) otherwise match the attention-name pattern and would be patched instead
    of the real attention.
    """
    if isinstance(module, Quantizer) or "Quantizer" in type(module).__name__:
        return True
    return any(segment.endswith("_quantizer") for segment in name.split("."))


def _is_kv_quantizable(name: str, module: nn.Module) -> bool:
    """Selection pattern (documented on the module docstring): class or name
    path marks attention; quantizer-owned submodules never qualify."""
    if _is_quant_internal(name, module):
        return False
    if isinstance(module, nn.Linear):
        return False
    cls_name = type(module).__name__
    if "Attention" in cls_name or "MLA" in cls_name:
        return True
    lowered = name.lower()
    return "self_attn" in lowered or "attention" in lowered


def apply_kv_quantization(
    model: nn.Module, kv_cfg: KVQConfig | None, kv_amax_ema_decay: float = 0.999
) -> list[KVQuantHandle]:
    """Attach KV-cache quantization simulation to every matching attention
    module (§6.2).

    Mechanism: for each selected module the K/V entry point is patched once — forward-argument
    interception when ``forward`` declares K/V-named parameters, otherwise projection-boundary
    interception on ``k_proj``/``v_proj`` (or the fused ``kv_b_proj`` with independent K/V scale
    sets). Each module gets one :class:`KVCacheQuantizer` stored as a private attribute whose
    buffers are non-persistent, so ``state_dict`` is unchanged. ``kv_cfg=None`` (KV cache stays
    bf16, §6.4) is a no-op returning ``[]``.

    Idempotency: re-applying to an already-wrapped module returns the existing handle instead of
    stacking a second quantization.

    Args:
        model (nn.Module): Model (or a bare attention module) to walk with ``named_modules()``.
        kv_cfg (KVQConfig | None): KV cache config; ``None`` disables the simulation entirely.
        kv_amax_ema_decay (float): EMA decay forwarded to each :class:`KVCacheQuantizer` (§6.3).

    Returns:
        list[KVQuantHandle]: One handle per wrapped attention module; ``handle.detach()`` restores
        the module. Empty when ``kv_cfg`` is ``None`` or no module matches.
    """
    if kv_cfg is None:
        return []
    handles: list[KVQuantHandle] = []
    for name, module in list(model.named_modules()):
        if not _is_kv_quantizable(name, module):
            continue
        existing = getattr(module, _HANDLE_ATTR, None)
        if isinstance(existing, KVQuantHandle):
            handles.append(existing)
            continue
        quantizer = KVCacheQuantizer(ema_decay=kv_amax_ema_decay)
        handle = KVQuantHandle(module, quantizer)
        if not _patch_forward_args(module, quantizer, handle):
            _patch_projections(module, quantizer, handle)
        setattr(module, _QUANTIZER_ATTR, quantizer)
        setattr(module, _HANDLE_ATTR, handle)
        handles.append(handle)
    return handles


def remove_kv_quantization(model: nn.Module) -> list[KVQuantHandle]:
    """Detach every KV quantization handle previously attached under ``model``
    (§5.4 symmetry).

    Args:
        model (nn.Module): Model whose wrapped attention modules are restored in place.

    Returns:
        list[KVQuantHandle]: The handles that were detached (in ``named_modules()`` order).
    """
    detached: list[KVQuantHandle] = []
    for _name, module in list(model.named_modules()):
        handle = getattr(module, _HANDLE_ATTR, None)
        if isinstance(handle, KVQuantHandle):
            handle.detach()
            detached.append(handle)
    return detached


__all__ = ["KVCacheQuantizer", "KVQuantHandle", "apply_kv_quantization", "remove_kv_quantization"]
