"""Sanity checks for the shared helpers."""

from __future__ import annotations

import pytest
import torch

from common.device import device_report, get_device
from common.shapes import assert_shape, show
from common.timing import benchmark


def test_device_is_usable(device: torch.device) -> None:
    x = torch.randn(4, 4, device=device)
    assert (x @ x).shape == (4, 4)


def test_device_report_mentions_torch() -> None:
    assert "torch" in device_report()


def test_assert_shape_passes_on_match() -> None:
    t = torch.zeros(2, 4, 8)
    assert assert_shape(t, (2, 4, 8), "t") is t


def test_assert_shape_names_the_tensor_on_failure() -> None:
    t = torch.zeros(2, 4, 8)
    with pytest.raises(AssertionError, match=r"scores: expected shape \(2, 4, 4\)"):
        assert_shape(t, (2, 4, 4), "scores")


def test_show_returns_its_input() -> None:
    t = torch.zeros(3, 3)
    assert show("t", t) is t


def test_benchmark_reports_timings(device: torch.device) -> None:
    x = torch.randn(64, 64, device=device)
    result = benchmark(lambda: x @ x, device, warmup=2, iters=5)
    assert result["median_ms"] > 0
    assert result["p90_ms"] >= result["min_ms"]
