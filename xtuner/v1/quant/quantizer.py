"""Integer symmetric fake-quantizer for QAD training (design doc §4.3,
§5.1)."""

from typing import Literal

import torch


_EPS_FP32: float = torch.finfo(torch.float32).eps
_SCOPES: tuple[str, ...] = ("per_channel", "per_group", "per_token", "per_batch")


class Quantizer(torch.nn.Module):
    """Dtype-agnostic fake-quant protocol (design doc §5.1).

    Concrete implementations (e.g. :class:`IntQuantizer`) compute scales from the incoming
    tensor and fake-quantize it with a straight-through estimator. The base class registers no
    parameters or buffers, so nothing leaks into ``state_dict``.

    Args:
        dtype (str): Quantized dtype, e.g. ``"int8"`` or ``"int4"``.
        scope (str): Quantization scope, e.g. ``"per_channel"`` or ``"per_token"``.
    """

    def __init__(self, dtype: str, scope: str) -> None:
        super().__init__()
        self.dtype: str = dtype
        self.scope: str = scope

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the quantization scale of ``x`` under this quantizer's
        scope.

        Args:
            x (torch.Tensor): Incoming tensor (weight or activation).

        Returns:
            torch.Tensor: fp32 scale, broadcastable to ``x``.
        """
        raise NotImplementedError

    def quant_dequant(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Fake-quantize ``x`` with ``scale`` and dequantize back to ``x``'s
        dtype.

        Args:
            x (torch.Tensor): Incoming tensor (weight or activation).
            scale (torch.Tensor): Scale from :meth:`compute_scale`, broadcastable to ``x``.

        Returns:
            torch.Tensor: Dequantized tensor with the same dtype and shape as ``x``.
        """
        raise NotImplementedError


class _QuantDequantSTE(torch.autograd.Function):
    """Forward = deploy-IR numerics; backward = identity with a ``|x| <= amax``
    clip mask."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.BackwardCFunction,
        x: torch.Tensor,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        x_f32 = x.detach().float()
        ctx.save_for_backward((x_f32.abs() <= scale * qmax).to(torch.float32))
        q = torch.clamp(torch.round(x_f32 / scale), qmin, qmax)
        return (q * scale).to(x.dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.BackwardCFunction, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        (mask,) = ctx.saved_tensors
        return grad_outputs[0] * mask, None, None, None, None


class IntQuantizer(Quantizer):
    """Symmetric integer fake-quantizer for int8/int4 (design doc §4.3).

    All scale/quantize math runs in float32; the dequantized output is cast back to the
    incoming tensor's dtype. No parameters or buffers are registered.

    Args:
        bits (Literal[8, 4] | int): Quantized bit width; only 8 and 4 are supported.
        scope (str): One of ``"per_channel"`` (weights, per output channel), ``"per_group"``
            (weights, groups along the input-channel dim), ``"per_token"`` (activations,
            reduced over the hidden dim) or ``"per_batch"`` (activations, reduced over the
            batch dim; W4A8 deployment quirk §4.3.3).
        group_size (int | None): Group size for ``"per_group"``; must be ``None`` otherwise.
    """

    def __init__(self, bits: Literal[8, 4] | int, scope: str, group_size: int | None = None) -> None:
        if bits not in (4, 8):
            raise ValueError(f"IntQuantizer: unsupported bits={bits}; only 4 and 8 are supported")
        if scope not in _SCOPES:
            raise ValueError(f"IntQuantizer: unknown scope={scope!r}; expected one of {_SCOPES}")
        if scope == "per_group" and (group_size is None or group_size <= 0):
            raise ValueError(f"IntQuantizer: scope='per_group' requires a positive group_size, got {group_size}")
        if scope != "per_group" and group_size is not None:
            raise ValueError(f"IntQuantizer: group_size must be None for scope={scope!r}, got {group_size}")
        super().__init__(dtype="int8" if bits == 8 else "int4", scope=scope)

        self.bits: int = int(bits)
        self.group_size: int | None = group_size
        self.qmin: int = -(2 ** (bits - 1))
        self.qmax: int = 2 ** (bits - 1) - 1

    def _repr_suffix(self) -> str:
        if self.group_size is None:
            return f"bits={self.bits}, scope='{self.scope}'"
        return f"bits={self.bits}, scope='{self.scope}', group_size={self.group_size}"

    def _amax(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-contained symmetric amax of ``x`` in fp32 under this
        quantizer's scope."""
        x_f32 = x.detach().float()
        if self.scope == "per_channel":
            if x_f32.dim() <= 1:
                return torch.maximum(x_f32.min().clamp(max=0).neg(), x_f32.max().clamp(min=0)).reshape([1])
            flat = x_f32.reshape(x_f32.shape[0], -1)
            return torch.maximum(flat.min(dim=1).values.clamp(max=0).neg(), flat.max(dim=1).values.clamp(min=0))
        if self.scope == "per_group":
            group_size = self.group_size
            assert group_size is not None
            grouped = x_f32.reshape(-1, group_size)
            return torch.maximum(grouped.min(dim=1).values.clamp(max=0).neg(), grouped.max(dim=1).values.clamp(min=0))
        # Activation scopes: reshape (-1, H); per_token reduces dim=1, per_batch reduces dim=0.
        x2d = x_f32.reshape(-1, x_f32.shape[-1])
        if self.scope == "per_token":
            amax = torch.maximum(x2d.min(dim=1).values.clamp(max=0).neg(), x2d.max(dim=1).values.clamp(min=0))
            return amax.unsqueeze(-1)
        return torch.maximum(x2d.min(dim=0).values.clamp(max=0).neg(), x2d.max(dim=0).values.clamp(min=0))

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fp32 symmetric scales of ``x`` (zero-containment, eps floor,
        §4.3).

        Args:
            x (torch.Tensor): Incoming tensor. For ``"per_channel"`` a weight of shape
                ``(out_features, in_features)`` (or any tensor whose trailing dims are reduced
                per leading channel); for ``"per_group"`` a 2-D weight; for activation scopes
                a tensor whose last dim is the hidden dim.

        Returns:
            torch.Tensor: fp32 scale broadcastable to ``x``: ``(out_features,)`` per-channel,
            ``(out_features, in_features // group_size)`` per-group, ``(N, 1)`` per-token or
            ``(H,)`` per-batch.
        """
        if x.numel() == 0:
            raise ValueError(f"IntQuantizer({self._repr_suffix()}): cannot compute scale of an empty tensor")
        level = 2 ** (self.bits - 1) - 1
        if self.scope == "per_group":
            if x.dim() != 2:
                raise ValueError(
                    f"IntQuantizer({self._repr_suffix()}): per-group scope expects a 2-D weight, got shape "
                    f"{tuple(x.shape)}"
                )
            group_size = self.group_size
            assert group_size is not None
            if x.shape[1] % group_size != 0:
                raise ValueError(
                    f"IntQuantizer({self._repr_suffix()}): in_features={x.shape[1]} is not divisible by "
                    f"group_size={group_size}; msmodelslim does not pad per-group scales"
                )
            amax = self._amax(x).reshape(x.shape[0], x.shape[1] // group_size)
            return torch.clamp(amax / level, min=_EPS_FP32)
        return torch.clamp(self._amax(x) / level, min=_EPS_FP32)

    def quant_dequant(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Fake-quantize ``x`` with STE (§4.3.8); forward matches the deploy IR
        exactly.

        Args:
            x (torch.Tensor): Incoming tensor; bf16/fp16 inputs are computed in fp32 and the
                result cast back to this dtype.
            scale (torch.Tensor): Scale from :meth:`compute_scale`.

        Returns:
            torch.Tensor: ``clamp(round_half_even(x / scale), qmin, qmax) * scale`` in ``x``'s
            dtype; backward passes gradients through in-range positions and zeroes them where
            ``|x| > amax``.
        """
        scale_r = scale.float()
        if self.scope == "per_group":
            assert self.group_size is not None
            x_r = x.reshape(x.shape[0], -1, self.group_size)
            scale_r = scale_r.reshape(x.shape[0], -1, 1)
            return _QuantDequantSTE.apply(x_r, scale_r, self.qmin, self.qmax).reshape(x.shape)
        if self.scope == "per_channel" and scale.dim() == 1 and x.dim() > 1:
            scale_r = scale_r.reshape([x.shape[0]] + [1] * (x.dim() - 1))
        return _QuantDequantSTE.apply(x, scale_r, self.qmin, self.qmax)


__all__ = ["IntQuantizer", "Quantizer"]
