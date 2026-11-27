import os

import torch

from xtuner.v1.utils import get_device, log_rank0

from .protocol import FlashAttnVarlenProtocol, cpu_flash_varlen_attn


def _fa2_available() -> bool:
    try:
        from flash_attn.flash_attn_interface import flash_attn_gpu  # noqa: F401
    except ImportError:
        return False
    return True


def _fa3_entry_available() -> bool:
    try:
        from flash_attn_interface import flash_attn_3_gpu  # noqa: F401
    except ImportError:
        try:
            from flash_attn_interface import flash_attn_3_cuda  # noqa: F401
        except ImportError:
            return False
    return True


def _fa3_arch_supported() -> bool:
    """Return whether the CUDA device can run FlashAttention-3 kernels.

    FlashAttention-3 kernels are built for Hopper (sm90) and newer. On older
    architectures the dist still imports and dispatches, but produces numerically
    wrong results (validated on sm86: output uncorrelated with the fp64 reference),
    so the architecture must be probed before selecting FA3.

    Returns:
        bool: True when the current CUDA device major compute capability is >= 9.
    """
    return torch.cuda.get_device_capability()[0] >= 9


def _unavailable_flash_attn(*args: object, **kwargs: object) -> FlashAttnVarlenProtocol:
    raise RuntimeError(
        "FlashAttention callables are unavailable on this device (flash-attn not installed and/or XTUNER_USE_FA3 "
        "set on a device without usable FA3 kernels). Set `attn_impl='flex_attention'` (the default fallback) or "
        "install flash-attn."
    )


def get_flash_attn_varlen() -> FlashAttnVarlenProtocol:
    """Resolve the flash attention varlen callable for the current device.

    Selection probes the actual callables instead of trusting package installation:
    an installed `flash-attn-3` dist does not imply usable FA2/FA3 kernels. This
    never returns ``None``; when no flash backend is usable it returns a stub that
    raises a descriptive ``RuntimeError``, so callers must gate on
    :func:`flash_attn_available`.

    With ``XTUNER_USE_FA3=1`` (or ``true``) an explicitly requested but missing FA3
    entry point raises ``RuntimeError`` (fail fast on an explicit request), while FA3
    kernels that exist but cannot run on this GPU (pre-Hopper) are skipped with a
    warning and the FA2/stub fallback applies (fail soft on unusable kernels).

    Returns:
        FlashAttnVarlenProtocol: The selected backend callable, the CPU/NPU
        placeholder, or the unavailable stub.
    """
    device = get_device()
    if device == "cpu":
        return cpu_flash_varlen_attn

    if device == "npu":
        from .npu import npu_flash_varlen_attn

        return npu_flash_varlen_attn

    if device != "cuda":
        raise RuntimeError(f"Unsupported device for flash attention: {device}")

    # TODO: #  Uniofy "1" and "true"
    # For the optional feature, we should collect all information to logs
    if os.environ.get("XTUNER_USE_FA3", "0").strip().lower() in {"1", "true"}:
        if not _fa3_entry_available():
            raise RuntimeError(
                "XTUNER_USE_FA3 is set but FlashAttention-3 is not importable; install flash-attn-3 or unset "
                "XTUNER_USE_FA3."
            )
        if _fa3_arch_supported():
            from .gpu import gpu_flash_varlen_attn_v3

            return gpu_flash_varlen_attn_v3
        major, minor = torch.cuda.get_device_capability()
        log_rank0.warning(
            f"XTUNER_USE_FA3 is set but FlashAttention-3 kernels require Hopper (sm90+); this device is "
            f"sm{major}{minor}. Ignoring the flag and falling back."
        )
    if _fa2_available():
        from .gpu import flash_attn_varlen_func_v2

        return flash_attn_varlen_func_v2
    log_rank0.warning(
        "flash-attn (FA2) is not installed and no usable FA3 backend was selected; flash attention is unavailable "
        "on this device."
    )
    return _unavailable_flash_attn


def flash_attn_available() -> bool:
    """Return whether a usable flash attention backend can be resolved on this
    device.

    Catches the explicit-request ``RuntimeError`` raised by :func:`get_flash_attn_varlen`
    so that model construction can downgrade to ``flex_attention`` instead of crashing.

    Returns:
        bool: True when :func:`get_flash_attn_varlen` resolves a real backend callable.
    """
    try:
        return get_flash_attn_varlen() is not _unavailable_flash_attn
    except RuntimeError:
        return False


flash_attn_varlen_func = get_flash_attn_varlen()
