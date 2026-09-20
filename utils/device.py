"""
utils/device.py

Auto-detects the best available device:
  - Apple M2 / Apple Silicon → "mps"
  - NVIDIA GPU               → "cuda"
  - Fallback                 → "cpu"

Usage:
    from utils.device import get_device, to_device
    device = get_device()
    tensor = to_device(my_tensor, device)
"""

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """
    Returns the best available torch.device.

    prefer: "auto" | "mps" | "cuda" | "cpu"
    """
    if prefer != "auto":
        d = torch.device(prefer)
        _validate(d)
        return d

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon MPS (M1/M2/M3)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    return torch.device("cpu")


def _validate(device: torch.device):
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device.type == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("MPS not compiled in this PyTorch build.")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available on this machine.")


def device_info(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA — {name} ({mem:.1f} GB VRAM)"
    if device.type == "mps":
        import platform
        chip = platform.processor() or "Apple Silicon"
        return f"MPS — {chip} (unified memory)"
    return "CPU"


def to_device(obj, device: torch.device):
    """Move a tensor, dict of tensors, or list of tensors to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


# ── MPS-specific workarounds ──────────────────────────────────────────────────

def safe_gumbel_softmax(logits: torch.Tensor, tau: float = 1.0,
                        hard: bool = True, dim: int = -1) -> torch.Tensor:
    """
    F.gumbel_softmax on MPS occasionally produces NaN with hard=True.
    This wrapper falls back to CPU for the sampling and moves back to MPS.
    Remove the workaround once the upstream MPS bug is fixed.
    """
    import torch.nn.functional as F
    if logits.device.type == "mps":
        # Sample on CPU, move result back
        result = F.gumbel_softmax(logits.cpu(), tau=tau, hard=hard, dim=dim)
        return result.to(logits.device)
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=dim)


def mixed_precision_context(device: torch.device):
    """
    Returns the appropriate autocast context for the device.
    CUDA: torch.cuda.amp.autocast (fp16)
    MPS:  torch.autocast("cpu")  — MPS autocast is experimental; use bfloat16
    CPU:  no-op context
    """
    import contextlib
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    if device.type == "mps":
        # bfloat16 is stable on MPS from PyTorch 2.1
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()
