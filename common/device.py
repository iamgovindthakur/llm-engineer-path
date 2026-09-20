"""Device selection for an Apple-silicon box with no local CUDA."""

from __future__ import annotations

import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """Return the best available device.

    Order: MPS (Apple GPU) -> CUDA (if you are on a rented box) -> CPU.

    Pass ``prefer_mps=False`` to force CPU. That is worth doing whenever a
    result looks numerically wrong: MPS accumulates in different orders than
    CPU, and a handful of ops silently fall back. If CPU and MPS disagree
    beyond ``1e-4``, trust CPU while you are learning.
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report() -> str:
    """One-line summary of what this machine can actually do.

    Print this at the top of a notebook so a surprising result later can be
    traced back to the hardware rather than to the math.
    """
    parts = [f"torch {torch.__version__}"]
    parts.append(f"mps_available={torch.backends.mps.is_available()}")
    parts.append(f"mps_built={torch.backends.mps.is_built()}")
    parts.append(f"cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        parts.append(f"gpu={torch.cuda.get_device_name(0)}")
    parts.append(f"selected={get_device()}")
    return " | ".join(parts)


if __name__ == "__main__":
    print(device_report())
