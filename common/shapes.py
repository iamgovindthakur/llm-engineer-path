"""Shape inspection helpers.

Most PyTorch bugs at this stage are shape bugs that never raise: broadcasting
turns a mistake into a plausible-looking tensor. Printing shapes constantly is
the cheapest defence there is.
"""

from __future__ import annotations

from typing import Iterable

import torch


def show(name: str, t: torch.Tensor, values: bool = False) -> torch.Tensor:
    """Print a tensor's shape (and optionally its values), then return it.

    Returns the tensor so it can be dropped into an expression without
    restructuring the code::

        scores = show("scores", Q @ K.transpose(-2, -1))
    """
    line = f"{name:<24} shape={tuple(t.shape)}  dtype={t.dtype}  device={t.device}"
    print(line)
    if values:
        print(t)
    return t


def shape_table(tensors: dict[str, torch.Tensor]) -> None:
    """Print a markdown-ish table of several tensors at once."""
    width = max((len(k) for k in tensors), default=4)
    print(f"{'tensor'.ljust(width)} | shape")
    print(f"{'-' * width}-+------")
    for name, t in tensors.items():
        print(f"{name.ljust(width)} | {tuple(t.shape)}")


def assert_shape(t: torch.Tensor, expected: Iterable[int], name: str = "tensor") -> torch.Tensor:
    """Fail loudly and immediately when a shape is not what you predicted.

    Use this right after any reshape/transpose/head-split. Catching the wrong
    shape at the line that produced it is worth far more than catching a bad
    loss twenty lines later.
    """
    expected = tuple(expected)
    actual = tuple(t.shape)
    if actual != expected:
        raise AssertionError(f"{name}: expected shape {expected}, got {actual}")
    return t
