from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass(slots=True)
class WelfordVec:
    """
    Online mean / M2 accumulator for vectors.

    We store M2 per-coordinate (same shape as mean) so that
    sum(var_i) == M2.sum()/(n-1) matches the manuscript's use of
    V_{l,k} = sum_i Var(Delta^{(i)}_{l,k}).
    """

    mean: np.ndarray
    m2: np.ndarray
    n: int = 0

    @classmethod
    def zeros(cls, dim: int, dtype=np.float64) -> "WelfordVec":
        mean = np.zeros(dim, dtype=dtype)
        m2 = np.zeros(dim, dtype=dtype)
        return cls(mean=mean, m2=m2, n=0)

    def update_batch(self, x: np.ndarray) -> None:
        """
        Update with a batch x of shape (m, d).
        """
        if x.size == 0:
            return
        if x.ndim != 2:
            raise ValueError(f"Expected x to have ndim=2, got shape {x.shape}")
        m = x.shape[0]
        if m == 0:
            return

        # Batch Welford update (Chan et al. style)
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)

        if self.n == 0:
            self.mean[...] = batch_mean
            self.m2[...] = batch_m2
            self.n = m
            return

        n_a = self.n
        n_b = m
        delta = batch_mean - self.mean
        n = n_a + n_b
        self.mean[...] = self.mean + delta * (n_b / n)
        self.m2[...] = self.m2 + batch_m2 + (delta**2) * (n_a * n_b / n)
        self.n = n

    @property
    def var_sum(self) -> float:
        if self.n <= 1:
            return float("inf")
        return float(self.m2.sum() / (self.n - 1))


@dataclass(slots=True)
class ResamplingAcc:
    """
    Maintain partitioned sums/counts to compute leave-one-partition-out means
    efficiently (O(re_part * d) to materialize all LOO means).
    """

    re_part: int
    sum_total: np.ndarray  # shape (d,)
    cnt_total: int
    sum_part: np.ndarray  # shape (re_part, d)
    cnt_part: np.ndarray  # shape (re_part,)

    @classmethod
    def zeros(cls, re_part: int, dim: int, dtype=np.float64) -> "ResamplingAcc":
        return cls(
            re_part=re_part,
            sum_total=np.zeros(dim, dtype=dtype),
            cnt_total=0,
            sum_part=np.zeros((re_part, dim), dtype=dtype),
            cnt_part=np.zeros(re_part, dtype=np.int64),
        )

    def update_batch(self, x: np.ndarray) -> None:
        """
        x: shape (m, d)
        """
        if x.size == 0:
            return
        m = x.shape[0]
        start = self.cnt_total
        idxs = (np.arange(start, start + m) % self.re_part).astype(np.int64)

        self.sum_total += x.sum(axis=0)
        self.cnt_total += m

        # Update per-partition sums/counts
        for p in range(self.re_part):
            mask = idxs == p
            c = int(mask.sum())
            if c:
                self.cnt_part[p] += c
                self.sum_part[p] += x[mask].sum(axis=0)

    def loo_means(self) -> np.ndarray:
        """
        Returns array of shape (re_part, d), where row p is the mean excluding
        samples that fell into partition p.
        """
        d = self.sum_total.shape[0]
        out = np.empty((self.re_part, d), dtype=self.sum_total.dtype)
        for p in range(self.re_part):
            denom = self.cnt_total - int(self.cnt_part[p])
            if denom <= 0:
                # Fallback: if we can't exclude, use full mean
                out[p] = self.sum_total / max(self.cnt_total, 1)
            else:
                out[p] = (self.sum_total - self.sum_part[p]) / denom
        return out


@dataclass(slots=True)
class LevelState:
    """
    Statistics for a single level (either base gradient or a gradient difference).

    - base level: Delta = grad(x_l)
    - diff level: Delta = grad(x_l) - grad(x_prev)
    """

    x: np.ndarray
    cost: int  # 1 for base, 2 for diff
    x_prev: Optional[np.ndarray]  # None for base
    sample_fn: Callable[[int], Any]
    delta_stats: WelfordVec
    base_stats: Optional[WelfordVec]  # only for diff levels (for restart/clipping heuristics)
    m_min: int
    # Optional resampling accumulators (enabled when resampling is enabled)
    delta_resamp: Optional[ResamplingAcc] = None
    base_resamp: Optional[ResamplingAcc] = None

    # Bookkeeping for bias/stat error decomposition (matches current implementation style)
    m_prev: int = 0

    @property
    def m(self) -> int:
        return self.delta_stats.n

    @property
    def v_delta(self) -> float:
        return self.delta_stats.var_sum

    @property
    def v_base(self) -> float:
        if self.base_stats is None:
            return self.delta_stats.var_sum
        return self.base_stats.var_sum

    @property
    def v_batch(self) -> float:
        """
        Matches v1's notion:
        - base level: v_batch is variance of grad(x_0)
        - diff level: v_batch is variance of grad(x_l) (the "top" gradient)
        """
        return self.v_base

    @property
    def mean_delta(self) -> np.ndarray:
        return self.delta_stats.mean

    def delta_loo_means(self) -> Optional[np.ndarray]:
        if self.delta_resamp is None:
            return None
        return self.delta_resamp.loo_means()
