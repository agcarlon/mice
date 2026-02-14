"""
TimedMICE: MICE subclass that accumulates wall-clock time for overhead breakdown.

Does not modify mice; only subclasses and overrides methods to wrap with
time.perf_counter(). Used to report:
  (i)   variance_sizing  — _get_opt_ml, _get_opt_ml_for_levels
  (ii)  resampling       — _define_tol_resampling (and _define_tol when resampling)
  (iii) index_set        — _check_dropping, _check_restart, _check_clipping, _add_level
       Sub-keys for index-set operator breakdown:
         index_set_add, index_set_drop, index_set_restart, index_set_clip
  (iv)  gradient         — all grad(x, theta) calls (via wrapped grad in __post_init__)
"""

from __future__ import annotations

import time
from typing import Any, Callable, List

import numpy as np

from mice.core_impl import MICE
from mice.core_impl import MICE as _MICEImpl  # for _get_opt_ml_for_levels etc.
from mice.state import LevelState


def _timed(key: str, timing: dict[str, float]):
    """Context manager that adds elapsed time to timing[key]."""

    class _Ctx:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *args):
            timing[key] = timing.get(key, 0.0) + (time.perf_counter() - self.t0)
            return None

    return _Ctx()


class TimedMICE(MICE):
    """
    MICE that records cumulative seconds in:
      .timing['variance_sizing']  — opt sample-size computation (uses variance)
      .timing['resampling']       — resampling for tol/stopping
      .timing['index_set']        — total of add/drop/restart/clip (for first plot)
      .timing['index_set_add']    — _add_level only
      .timing['index_set_drop']   — _check_dropping only
      .timing['index_set_restart']— _check_restart only
      .timing['index_set_clip']   — _check_clipping only
      .timing['gradient']         — gradient oracle calls
    Call .reset_timing() to zero counters; run as usual, then read .timing.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._timing: dict[str, float] = {
            "variance_sizing": 0.0,
            "resampling": 0.0,
            "index_set": 0.0,
            "index_set_add": 0.0,
            "index_set_drop": 0.0,
            "index_set_restart": 0.0,
            "index_set_clip": 0.0,
            "gradient": 0.0,
        }
        _grad = self.grad

        def wrapped_grad(x: np.ndarray, thetas: Any) -> np.ndarray:
            t0 = time.perf_counter()
            out = _grad(x, thetas)
            self._timing["gradient"] = self._timing.get("gradient", 0.0) + (
                time.perf_counter() - t0
            )
            return out

        object.__setattr__(self, "grad", wrapped_grad)

    @property
    def timing(self) -> dict[str, float]:
        return self._timing

    def reset_timing(self) -> None:
        for k in self._timing:
            self._timing[k] = 0.0

    # --- Variance/sizing: opt_ml computation ---
    def _get_opt_ml(self, err_tol: float) -> np.ndarray:
        with _timed("variance_sizing", self._timing):
            return _MICEImpl._get_opt_ml(self, err_tol)

    def _get_opt_ml_for_levels(self, levels: List[LevelState], err_tol: float) -> np.ndarray:
        with _timed("variance_sizing", self._timing):
            return _MICEImpl._get_opt_ml_for_levels(self, levels, err_tol)

    # --- Resampling: tol and stopping quantiles (_define_tol calls this when use_resampling) ---
    def _define_tol_resampling(self) -> float:
        with _timed("resampling", self._timing):
            return _MICEImpl._define_tol_resampling(self)

    # --- Index-set: per-operator timing (each also contributes to index_set total) ---
    def _add_level(self, x: np.ndarray) -> None:
        t0 = time.perf_counter()
        out = _MICEImpl._add_level(self, x)
        dt = time.perf_counter() - t0
        self._timing["index_set_add"] = self._timing.get("index_set_add", 0.0) + dt
        self._timing["index_set"] = self._timing.get("index_set", 0.0) + dt
        return out

    def _check_dropping(self, opt_ml: np.ndarray, err_tol: float):
        t0 = time.perf_counter()
        out = _MICEImpl._check_dropping(self, opt_ml, err_tol)
        dt = time.perf_counter() - t0
        self._timing["index_set_drop"] = self._timing.get("index_set_drop", 0.0) + dt
        self._timing["index_set"] = self._timing.get("index_set", 0.0) + dt
        return out

    def _check_restart(self, opt_ml: np.ndarray, err_tol: float):
        t0 = time.perf_counter()
        out = _MICEImpl._check_restart(self, opt_ml, err_tol)
        dt = time.perf_counter() - t0
        self._timing["index_set_restart"] = self._timing.get("index_set_restart", 0.0) + dt
        self._timing["index_set"] = self._timing.get("index_set", 0.0) + dt
        return out

    def _check_clipping(self, opt_ml: np.ndarray, err_tol: float):
        t0 = time.perf_counter()
        out = _MICEImpl._check_clipping(self, opt_ml, err_tol)
        dt = time.perf_counter() - t0
        self._timing["index_set_clip"] = self._timing.get("index_set_clip", 0.0) + dt
        self._timing["index_set"] = self._timing.get("index_set", 0.0) + dt
        return out
