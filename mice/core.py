"""Compatibility shim for core imports."""

from __future__ import annotations

from .core_impl import GradFn, MICE

__all__ = ["MICE", "GradFn"]
