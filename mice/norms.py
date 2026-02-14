from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class PlainNormEstimator:
    """
    Plain norm estimator: uses ||g_hat||.
    """

    convex: bool = False
    _best: float = float("inf")

    def update(self, g_hat: np.ndarray) -> float:
        n = float(np.linalg.norm(g_hat))
        if self.convex:
            self._best = min(self._best, n)
            return self._best
        return n


@dataclass(slots=True)
class ResamplingNormEstimator:
    """
    Resampling norm estimator: consumes a vector of resampled ||g_hat|| values
    and returns two quantiles:
    - a conservative low-quantile for error tolerance selection
    - a quantile for the stochastic stopping rule
    """

    re_part: int = 5
    re_quantile: float = 0.05
    stop_quantile: float = 0.95
    convex: bool = False
    _best_tol: float = float("inf")
    _best_stop: float = float("inf")

    def update_from_norms(self, norms: np.ndarray) -> tuple[float, float]:
        """
        norms: 1D array of ||g_hat^{(s)}|| values from resampling.
        Returns (tol_norm, stop_norm).
        """
        if norms.size == 0:
            raise ValueError("ResamplingNormEstimator.update_from_norms got empty norms array")
        tol = float(np.quantile(norms, self.re_quantile))
        stop = float(np.quantile(norms, self.stop_quantile))
        if self.convex:
            self._best_tol = min(self._best_tol, tol)
            self._best_stop = min(self._best_stop, stop)
            return self._best_tol, self._best_stop
        return tol, stop

