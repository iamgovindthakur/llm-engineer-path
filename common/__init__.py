"""Shared helpers for the LLM learning workspace.

Deliberately tiny. These exist so lessons do not repeat boilerplate, never to
hide an operation that is the point of the lesson.
"""

from .device import get_device, device_report
from .shapes import show, shape_table, assert_shape
from .timing import timed, benchmark

__all__ = [
    "get_device",
    "device_report",
    "show",
    "shape_table",
    "assert_shape",
    "timed",
    "benchmark",
]
