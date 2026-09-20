"""Shared pytest fixtures for from-scratch implementations.

The pattern these tests exist to support: you implement something by hand in a
lesson, and a test here asserts it matches PyTorch's reference implementation.
That assertion is what turns "my code ran" into "my code is correct", and it is
the same discipline you already apply to backend code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.device import get_device  # noqa: E402


@pytest.fixture(scope="session")
def device() -> torch.device:
    return get_device()


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Every test starts from the same random state."""
    torch.manual_seed(0)


def assert_matches_reference(
    mine: torch.Tensor,
    reference: torch.Tensor,
    name: str = "implementation",
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """Assert a hand-written tensor matches PyTorch's, with a readable failure.

    The tolerances are loose on purpose. MPS accumulates float32 in a different
    order than CPU, so bit-exact equality is the wrong bar. If this fails by
    more than a hair, re-run the comparison on CPU before suspecting the math --
    a device difference and a real bug look identical from the error message.
    """
    if mine.shape != reference.shape:
        raise AssertionError(
            f"{name}: shape mismatch -- yours {tuple(mine.shape)}, "
            f"reference {tuple(reference.shape)}"
        )
    max_diff = (mine - reference).abs().max().item()
    try:
        torch.testing.assert_close(mine, reference, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(
            f"{name}: values differ from the PyTorch reference "
            f"(max abs diff {max_diff:.3e}).\n{exc}"
        ) from None
