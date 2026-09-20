"""Honest timing on MPS.

GPU work is asynchronous. Timing it without synchronising measures how fast you
can enqueue work, not how fast it runs -- which is the single most common way
benchmarks in this space end up wrong.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Callable

import torch


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@contextmanager
def timed(label: str, device: torch.device):
    """Time a block, synchronising the device on both sides."""
    _sync(device)
    start = time.perf_counter()
    yield
    _sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"{label}: {elapsed_ms:.3f} ms")


def benchmark(
    fn: Callable[[], object],
    device: torch.device,
    warmup: int = 5,
    iters: int = 50,
) -> dict[str, float]:
    """Run ``fn`` repeatedly and report median and p90 milliseconds.

    Warmup matters: the first calls pay for kernel compilation and allocator
    growth. Median, not mean, because a single scheduling hiccup skews a mean
    and you care about the typical case. p90 is reported because in serving it
    is the tail that decides your SLO.
    """
    for _ in range(warmup):
        fn()
    _sync(device)

    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000)

    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": samples[int(0.9 * len(samples)) - 1],
        "min_ms": samples[0],
        "iters": float(iters),
    }
