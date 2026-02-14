from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .logging import Recorder
from .norms import PlainNormEstimator, ResamplingNormEstimator
from .policy import DropRestartClipPolicy
from .sampling import FiniteSampler, SamplerLike
from .state import LevelState, ResamplingAcc, WelfordVec


GradFn = Callable[[np.ndarray, Any], np.ndarray]


@dataclass
class MICE:
    """
    Implementation of the MICE estimator.

    NOTE: This file holds the single source of truth for MICE.
    """

    grad: GradFn
    sampler: SamplerLike
    eps: float = 0.577
    min_batch: int = 10
    restart_factor: int = 10
    max_cost: float = float("inf")
    stop_crit_norm: float = 0.0
    stop_crit_prob: float = 0.05

    convex: bool = False
    policy: DropRestartClipPolicy = field(default_factory=DropRestartClipPolicy)
    recorder: Optional[Recorder] = None
    # - resampling ON by default
    # - clipping OFF by default
    use_resampling: bool = True

    # Resampling controls
    re_part: int = 5
    re_quantile: float = 0.05
    # Match v1/paper defaults & semantics
    re_tot_cost: float = 0.2
    re_min_n: int = 5
    re_max_samp: int = 1000

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng()
        self.finite = not callable(self.sampler)
        self.data_size = len(self.sampler) if self.finite else None

        self.levels: List[LevelState] = []
        self.dim: Optional[int] = None

        self.counter = 0  # gradient evaluations
        self.k = 0  # estimator calls
        self.terminate = False
        self.terminate_reason: Optional[str] = None
        self.force_restart = False

        # For finite samplers, we keep a single FiniteSampler instance to maintain state
        # across levels, but we initialize it lazily
        self._finite_sampler: Optional[FiniteSampler] = None

        self.m_restart_min = self.restart_factor * self.min_batch
        if self.finite:
            self.m_restart_min = int(min(self.m_restart_min, self.data_size))

        # TODO this lext lines look awkward: make it as 
        # if self.use_resampling: self.norm_estimator = ... else self.norm_estimator = PlainNormEstimator...
        self.norm_estimator = (
            ResamplingNormEstimator(
                re_part=self.re_part,
                re_quantile=self.re_quantile,
                stop_quantile=self.stop_crit_prob,
                convex=self.convex,
            )
            if self.use_resampling
            else PlainNormEstimator(convex=self.convex)
        )
        self._norm_stop: Optional[float] = None
        # - err_tol is used inside the resampling routine to estimate the cost
        #   of a MICE iteration (via opt_ml - m_prev) before updating err_tol.
        self.err_tol: float = 1e-6 if self.use_resampling else 0.0
        self.re_cost: float = 1.0

        # Cached aggregate estimator: g_hat = sum(mean_delta_l)
        self._g_hat: Optional[np.ndarray] = None
        self._last_pilot_thetas: Any = None
        self._last_pilot_g_cur: Optional[np.ndarray] = None  # shape (m, d)

        if self.recorder is None:
            self.recorder = Recorder()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    # --- Public API ---
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate MICE at x, updating internal hierarchy.
        """
        if self.terminate:
            return np.full_like(np.asarray(x, dtype=float), np.nan)

        x = np.asarray(x, dtype=float).reshape(-1)
        if self.dim is None:
            self.dim = int(x.size)
        elif x.size != self.dim:
            raise ValueError(f"Inconsistent dimension: expected {self.dim}, got {x.size}")

        # Mark previous sample sizes for bias/stat-error decomposition
        for lvl in self.levels:
            lvl.m_prev = lvl.m

        current_event = "start" if len(self.levels) == 0 else "add"

        # Add a new level and take pilot samples (m_min)
        self._add_level(x)
        self._pilot_update_last_level()

        # Define tolerance on error using current norm estimate
        err_tol = self._define_tol()

        # Compute optimal sample sizes
        opt_ml = self._get_opt_ml(err_tol)

        # Policy decisions (drop/restart/clip) happen before sampling-more loop in v1
        if self.policy and len(self.levels) > 2:
            did_drop, opt_ml = self._check_dropping(opt_ml, err_tol)
            if did_drop:
                current_event = "dropped"
        if self.policy and len(self.levels) > 1:
            did_restart, opt_ml = self._check_restart(opt_ml, err_tol)
            if did_restart:
                current_event = "restart"
        did_clip, opt_ml = self._check_clipping(opt_ml, err_tol)
        if did_clip:
            current_event = "clip"

        # Sample more until sizes meet opt_ml
        while not self._check_samp_sizes(opt_ml):
            for lvl, m_opt in zip(self.levels, opt_ml):
                self._grow_level_to(lvl, int(m_opt))
                if self.terminate:
                    break
            if self.terminate:
                break
            err_tol = self._define_tol()
            opt_ml = self._get_opt_ml(err_tol)

        g_hat = self._g_hat if self._g_hat is not None else self._aggregate_recompute()

        # Stop criterion (v1 style)
        self._check_stop_crit(err_tol)
        if self.terminate:
            current_event = "end"

        # Log (exactly once per evaluate call)
        self._record(event=current_event, g_hat=g_hat)

        self.k += 1
        return g_hat

    def get_log(self):
        return self.recorder.as_list() if self.recorder else []

    # --- Internal helpers ---
    def _record(self, *, event: str, g_hat: np.ndarray) -> None:
        if not self.recorder:
            return
        grad_norm = float(np.linalg.norm(g_hat)) if g_hat is not None else None
        last_v = self.levels[-1].v_delta if self.levels else None
        self.recorder.add(
            event=event,
            num_grads=int(self.counter),
            hier_length=len(self.levels),
            last_v=last_v,
            grad_norm=grad_norm,
            iteration=int(self.k),
            terminate_reason=self.terminate_reason,
        )

    def _new_level_sampler(self):
        if not self.finite:
            return self.sampler  # callable
        # Reuse the single FiniteSampler instance to maintain state across levels.
        # Lazily initialize it using the current RNG so seeding works even if `rng`
        # is set after __post_init__ (common in experiment scripts).
        if self._finite_sampler is None:
            assert self.data_size is not None
            start = int(self.rng.integers(0, self.data_size))
            self._finite_sampler = FiniteSampler(data=self.sampler, start=start)
        return self._finite_sampler.next

    def _add_level(self, x: np.ndarray) -> None:
        if not self.levels:
            # base
            delta_stats = WelfordVec.zeros(self.dim)
            lvl = LevelState(
                x=x.copy(),
                x_prev=None,
                cost=1,
                sample_fn=self._new_level_sampler(),
                delta_stats=delta_stats,
                base_stats=None,
                m_min=self.m_restart_min,
                delta_resamp=ResamplingAcc.zeros(self.re_part, self.dim) if self.use_resampling else None,
                base_resamp=None,
            )
            self.levels.append(lvl)
            self._g_hat = np.zeros(self.dim)
            return

        prev = self.levels[-1].x
        delta_stats = WelfordVec.zeros(self.dim)
        base_stats = WelfordVec.zeros(self.dim)
        lvl = LevelState(
            x=x.copy(),
            x_prev=prev.copy(),
            cost=2,
            sample_fn=self._new_level_sampler(),
            delta_stats=delta_stats,
            base_stats=base_stats,
            m_min=self.min_batch,
            delta_resamp=ResamplingAcc.zeros(self.re_part, self.dim) if self.use_resampling else None,
            base_resamp=ResamplingAcc.zeros(self.re_part, self.dim) if self.use_resampling else None,
        )
        self.levels.append(lvl)

    def _pilot_update_last_level(self) -> None:
        """
        Take m_min samples for the last level and update its statistics.
        """
        lvl = self.levels[-1]
        m = int(lvl.m_min)
        if m <= 0:
            return
        thetas = lvl.sample_fn(m)
        if lvl.cost == 1:
            g = self.grad(lvl.x, thetas)
            if g.shape != (m, self.dim):
                raise ValueError(f"grad returned {g.shape}, expected {(m, self.dim)}")
            old_mean = lvl.mean_delta.copy()
            lvl.delta_stats.update_batch(g)
            if lvl.delta_resamp is not None:
                lvl.delta_resamp.update_batch(g)
            self._update_g_hat(old_mean, lvl.mean_delta)
            self.counter += m
            self._last_pilot_thetas = thetas
            self._last_pilot_g_cur = g
        else:
            g_cur = self.grad(lvl.x, thetas)
            g_prev = self.grad(lvl.x_prev, thetas)
            if g_cur.shape != (m, self.dim) or g_prev.shape != (m, self.dim):
                raise ValueError("grad returned wrong shape for diff level")
            old_mean = lvl.mean_delta.copy()
            lvl.base_stats.update_batch(g_cur)
            lvl.delta_stats.update_batch(g_cur - g_prev)
            if lvl.base_resamp is not None:
                lvl.base_resamp.update_batch(g_cur)
            if lvl.delta_resamp is not None:
                lvl.delta_resamp.update_batch(g_cur - g_prev)
            self._update_g_hat(old_mean, lvl.mean_delta)
            self.counter += 2 * m
            self._last_pilot_thetas = thetas
            self._last_pilot_g_cur = g_cur

    def _update_g_hat(self, old_mean: np.ndarray, new_mean: np.ndarray) -> None:
        if self._g_hat is None:
            self._g_hat = np.zeros(self.dim)
        self._g_hat += (new_mean - old_mean)

    def _aggregate_recompute(self) -> np.ndarray:
        self._g_hat = np.zeros(self.dim)
        for lvl in self.levels:
            self._g_hat += lvl.mean_delta
        return self._g_hat

    # --- Error control, sample sizes, and stopping ---
    def _define_tol(self) -> float:
        if self.use_resampling:
            self.err_tol = self._define_tol_resampling()
            return float(self.err_tol)
        g_hat = self._g_hat if self._g_hat is not None else self._aggregate_recompute()
        n = float(self.norm_estimator.update(g_hat))
        # v1 plain mode: err_tol = eps * ||g_hat||
        self.err_tol = float(self.eps * n)
        return float(self.err_tol)

    def _define_tol_resampling(self) -> float:
        """
        v1-style: err_tol = eps/(1+eps) * q_{re_quantile}( ||g_hat^{(res)}|| )
        and store stop-quantile for the stochastic stopping rule.
        """
        if not self.levels:
            self._norm_stop = 0.0
            return float(self.eps * 0.0)
        g_hat = self._g_hat if self._g_hat is not None else self._aggregate_recompute()
        L = len(self.levels)

        # Materialize leave-one-partition-out means per level
        loo_means = []
        for lvl in self.levels:
            arr = lvl.delta_loo_means()
            if arr is None:
                # Shouldn't happen if use_resampling=True, but keep safe fallback
                arr = np.tile(lvl.mean_delta[None, :], (self.re_part, 1))
            loo_means.append(arr)

        # v1/paper resampling sample size:
        #   re_samp = int(re_tot_cost * cost / (re_cost * |L|))
        #   cost = sum(max(opt_ml(err_tol_prev) - m_prev, 0))
        #   re_samp = min(re_samp, re_max_samp, (2*re_part)^|L|)
        #   n = max(re_samp, re_min_n)
        #
        # We use `self.err_tol` (previous tolerance) to estimate opt_ml,
        # matching v1's sequencing.
        ml_prev = np.asarray([lvl.m_prev for lvl in self.levels], dtype=float)
        opt_ml_prev = self._get_opt_ml(float(self.err_tol))
        cost = float(np.maximum(opt_ml_prev - ml_prev, 0.0).sum())
        re_samp = int(self.re_tot_cost * cost / (self.re_cost * max(L, 1)))
        re_samp = int(
            min(
                re_samp,
                int(self.re_max_samp),
                int((2 * self.re_part) ** L),
            )
        )
        n_samp = int(max(re_samp, int(self.re_min_n)))
        choices = self.rng.integers(0, self.re_part, size=(n_samp, L), dtype=np.int64)
        g_samp = np.zeros((n_samp, self.dim), dtype=float)
        for j in range(L):
            g_samp += loo_means[j][choices[:, j]]
        norms = np.linalg.norm(g_samp, axis=1)
        # Include the full estimator norm as an additional sample (matches v1 style)
        norms = np.concatenate([norms, np.asarray([float(np.linalg.norm(g_hat))])], axis=0)

        tol_norm, stop_norm = self.norm_estimator.update_from_norms(norms)  # type: ignore[attr-defined]
        self._norm_stop = float(stop_norm)
        return float((self.eps / (1.0 + self.eps)) * tol_norm)

    def _get_opt_ml(self, err_tol: float) -> np.ndarray:
        if self.finite:
            return self._get_opt_ml_finite(err_tol)
        return self._get_opt_ml_continuous(err_tol)

    def _get_opt_ml_continuous(self, err_tol: float) -> np.ndarray:
        vl = []
        ml = []
        cl = []
        m_min = []
        for i, lvl in enumerate(self.levels):
            vl.append(lvl.v_batch if i == 0 else lvl.v_delta)
            ml.append(lvl.m)
            cl.append(lvl.cost)
            m_min.append(self.m_restart_min if i == 0 else self.min_batch)
        vl = np.asarray(vl, dtype=float)
        ml = np.asarray(ml, dtype=float)
        cl = np.asarray(cl, dtype=float)
        m_min = np.asarray(m_min, dtype=float)

        constant = float(np.sum(np.sqrt(vl * cl)))
        opt_ml = np.ceil((err_tol ** (-2)) * np.sqrt(vl / cl) * constant).astype(int)
        opt_ml = np.maximum(opt_ml, m_min.astype(int))
        opt_ml = np.maximum(opt_ml, ml.astype(int))
        return opt_ml

    def _get_opt_ml_finite(self, err_tol: float) -> np.ndarray:
        assert self.data_size is not None
        ds = self.data_size

        vl = []
        ml = []
        cl = []
        m_min = []
        for i, lvl in enumerate(self.levels):
            vl.append(lvl.v_batch if i == 0 else lvl.v_delta)
            ml.append(lvl.m)
            cl.append(lvl.cost)
            m_min.append(self.m_restart_min if i == 0 else self.min_batch)

        vl = np.asarray(vl, dtype=float)
        ml = np.asarray(ml, dtype=float)
        cl = np.asarray(cl, dtype=float)
        m_min = np.asarray(m_min, dtype=int)

        opt_ml = m_min.copy().astype(int)
        ells = ml < ds
        while float(np.sum(vl / opt_ml * (1.0 - opt_ml / ds))) > float(err_tol**2):
            aux1 = float(err_tol**2) + (1.0 / ds) * float(np.sum(vl[ells]))
            aux2 = float(np.sum(np.sqrt(vl[ells] * cl[ells])))
            opt_ml[ells] = np.ceil(np.sqrt(vl[ells] / cl[ells]) * aux2 / aux1).astype(int)
            opt_ml = np.minimum(opt_ml, ds)
            opt_ml = np.maximum(opt_ml, ml.astype(int))
            opt_ml = np.maximum(opt_ml, m_min)
            ells = opt_ml < ds
        return opt_ml

    def _check_samp_sizes(self, opt_ml: np.ndarray) -> bool:
        return all(opt_ml <= np.asarray([lvl.m for lvl in self.levels], dtype=int))

    def _check_stop_crit(self, err_tol: float) -> None:
        if self.stop_crit_norm <= 0.0:
            return
        if self.use_resampling and self._norm_stop is not None:
            norm_est = float(self._norm_stop)
        else:
            g_hat = self._g_hat if self._g_hat is not None else self._aggregate_recompute()
            norm_est = float(np.linalg.norm(g_hat))
        l2_err = self._compute_error()
        if norm_est < np.sqrt(self.stop_crit_norm) - np.sqrt(max(l2_err, 0.0)):
            self.terminate = True
            self.terminate_reason = "stop_crit"

    def _compute_bias(self) -> float:
        if len(self.levels) <= 1:
            return 0.0
        bias = 0.0
        if self.finite:
            assert self.data_size is not None
            for lvl in self.levels[:-1]:
                factor = (self.data_size - lvl.m_prev) / self.data_size
                bias += factor * (lvl.m_prev / (lvl.m**2)) * lvl.v_delta
        else:
            for lvl in self.levels[:-1]:
                bias += (lvl.m_prev / (lvl.m**2)) * lvl.v_delta
        return float(bias)

    def _compute_statistical_error(self) -> float:
        if not self.levels:
            return 0.0
        stat_err = 0.0
        if self.finite:
            assert self.data_size is not None
            for lvl in self.levels[:-1]:
                factor = (self.data_size - lvl.m) / self.data_size
                stat_err += factor * ((lvl.m - lvl.m_prev) / (lvl.m**2)) * lvl.v_delta
            last = self.levels[-1]
            factor = (self.data_size - last.m) / self.data_size
            stat_err += factor * (last.v_delta / last.m)
        else:
            for lvl in self.levels[:-1]:
                stat_err += ((lvl.m - lvl.m_prev) / (lvl.m**2)) * lvl.v_delta
            last = self.levels[-1]
            stat_err += last.v_delta / last.m
        return float(stat_err)

    def _compute_error(self) -> float:
        return float(self._compute_bias() + self._compute_statistical_error())

    # --- Sampling growth ---
    def _grow_level_to(self, lvl: LevelState, m_opt: int) -> None:
        if lvl.m >= m_opt:
            return
        m_to_sample = min(m_opt - lvl.m, lvl.m if lvl.m > 0 else m_opt - lvl.m)
        m_min = lvl.m_min
        if m_to_sample <= 0:
            return
        if self.finite:
            assert self.data_size is not None
            m_to_sample = min(m_to_sample, self.data_size - lvl.m)
            m_min = min(m_min, self.data_size - lvl.m)
        if m_to_sample > 0:
            m_to_sample = max(m_to_sample, m_min)
        extra_eval = m_to_sample * lvl.cost
        if self.counter + extra_eval > self.max_cost:
            self.terminate = True
            self.terminate_reason = "max_cost"
            return

        thetas = lvl.sample_fn(int(m_to_sample))
        if lvl.cost == 1:
            g = self.grad(lvl.x, thetas)
            old_mean = lvl.mean_delta.copy()
            lvl.delta_stats.update_batch(g)
            if lvl.delta_resamp is not None:
                lvl.delta_resamp.update_batch(g)
            self._update_g_hat(old_mean, lvl.mean_delta)
            self.counter += int(m_to_sample)
        else:
            g_cur = self.grad(lvl.x, thetas)
            g_prev = self.grad(lvl.x_prev, thetas)
            old_mean = lvl.mean_delta.copy()
            lvl.base_stats.update_batch(g_cur)
            lvl.delta_stats.update_batch(g_cur - g_prev)
            if lvl.base_resamp is not None:
                lvl.base_resamp.update_batch(g_cur)
            if lvl.delta_resamp is not None:
                lvl.delta_resamp.update_batch(g_cur - g_prev)
            self._update_g_hat(old_mean, lvl.mean_delta)
            self.counter += int(2 * m_to_sample)

    # --- Policy: dropping / restart / clipping (v1-like) ---
    def _check_restart(self, opt_ml: np.ndarray, err_tol: float) -> Tuple[bool, np.ndarray]:
        ml = np.asarray([lvl.m for lvl in self.levels], dtype=float)
        mice_cost = float(np.maximum(0.0, np.ceil(opt_ml - ml)).sum() + self.policy.aggr_cost * len(opt_ml))

        new_delta = self._restart_delta(self.levels[-1])
        opt_ml_restart = self._get_opt_ml_for_levels([new_delta], err_tol)
        opt_ml_restart = np.maximum(opt_ml_restart, self.m_restart_min)
        restart_cost = float(np.maximum(0.0, opt_ml_restart[0] - ml[-1]) + self.policy.aggr_cost)

        if (
            restart_cost < mice_cost * (1.0 + self.policy.restart_param)
            or len(self.levels) > self.policy.max_hierarchy_size
            or self.force_restart
        ):
            self.force_restart = False
            self.levels = [new_delta]
            self._g_hat = new_delta.mean_delta.copy()
            return True, np.asarray(opt_ml_restart, dtype=int)
        return False, opt_ml

    def _restart_delta(self, last: LevelState) -> LevelState:
        if last.cost == 1:
            base_stats = last.delta_stats
        else:
            assert last.base_stats is not None
            base_stats = last.base_stats
        new_delta_stats = WelfordVec.zeros(self.dim)
        new_delta_stats.mean[...] = base_stats.mean
        new_delta_stats.m2[...] = base_stats.m2
        new_delta_stats.n = base_stats.n

        lvl = LevelState(
            x=last.x.copy(),
            x_prev=None,
            cost=1,
            sample_fn=last.sample_fn,
            delta_stats=new_delta_stats,
            base_stats=None,
            m_min=self.m_restart_min,
            m_prev=last.m_prev,
            delta_resamp=(last.delta_resamp if last.cost == 1 else last.base_resamp),
            base_resamp=None,
        )
        return lvl

    def _get_opt_ml_for_levels(self, levels: List[LevelState], err_tol: float) -> np.ndarray:
        if self.finite:
            assert self.data_size is not None
            ds = self.data_size
            vl = []
            ml = []
            cl = []
            m_min = []
            for i, lvl in enumerate(levels):
                vl.append(lvl.v_batch if i == 0 else lvl.v_delta)
                ml.append(lvl.m)
                cl.append(lvl.cost)
                m_min.append(self.m_restart_min if i == 0 else self.min_batch)
            vl = np.asarray(vl, dtype=float)
            ml = np.asarray(ml, dtype=float)
            cl = np.asarray(cl, dtype=float)
            m_min = np.asarray(m_min, dtype=int)

            opt_ml = m_min.copy().astype(int)
            ells = ml < ds
            while float(np.sum(vl / opt_ml * (1.0 - opt_ml / ds))) > float(err_tol**2):
                aux1 = float(err_tol**2) + (1.0 / ds) * float(np.sum(vl[ells]))
                aux2 = float(np.sum(np.sqrt(vl[ells] * cl[ells])))
                opt_ml[ells] = np.ceil(np.sqrt(vl[ells] / cl[ells]) * aux2 / aux1).astype(int)
                opt_ml = np.minimum(opt_ml, ds)
                opt_ml = np.maximum(opt_ml, ml.astype(int))
                opt_ml = np.maximum(opt_ml, m_min)
                ells = opt_ml < ds
            return opt_ml

        vl = []
        ml = []
        cl = []
        m_min = []
        for i, lvl in enumerate(levels):
            vl.append(lvl.v_batch if i == 0 else lvl.v_delta)
            ml.append(lvl.m)
            cl.append(lvl.cost)
            m_min.append(self.m_restart_min if i == 0 else self.min_batch)
        vl = np.asarray(vl, dtype=float)
        ml = np.asarray(ml, dtype=float)
        cl = np.asarray(cl, dtype=float)
        m_min = np.asarray(m_min, dtype=float)
        constant = float(np.sum(np.sqrt(vl * cl)))
        opt_ml = np.ceil((err_tol ** (-2)) * np.sqrt(vl / cl) * constant).astype(int)
        opt_ml = np.maximum(opt_ml, m_min.astype(int))
        opt_ml = np.maximum(opt_ml, ml.astype(int))
        return opt_ml

    def _check_dropping(self, opt_ml: np.ndarray, err_tol: float) -> Tuple[bool, np.ndarray]:
        ml = np.asarray([lvl.m for lvl in self.levels], dtype=float)
        mice_cost = float(np.maximum(0.0, np.ceil(opt_ml - ml)).sum() + self.policy.aggr_cost * len(opt_ml))

        delta_drop = self._build_drop_delta()
        if delta_drop is None:
            return False, opt_ml

        levels_drop = self.levels[:-2] + [delta_drop]
        opt_ml_drop = self._get_opt_ml_for_levels(levels_drop, err_tol)
        ml_drop = np.asarray([lvl.m for lvl in self.levels[:-2]] + [self.levels[-1].m], dtype=float)
        drop_cost = float(np.maximum(0.0, np.ceil(opt_ml_drop - ml_drop)).sum() + self.policy.aggr_cost * len(opt_ml_drop))

        if drop_cost <= mice_cost * (1.0 + self.policy.drop_param):
            self.levels = levels_drop
            self._g_hat = self._aggregate_recompute()
            return True, np.asarray(opt_ml_drop, dtype=int)
        return False, opt_ml

    def _build_drop_delta(self) -> Optional[LevelState]:
        if len(self.levels) < 3:
            return None
        x_k = self.levels[-1].x
        x_km2 = self.levels[-3].x

        m = int(self.min_batch)
        if self._last_pilot_thetas is not None and self._last_pilot_g_cur is not None:
            thetas = self._last_pilot_thetas
            g_cur = self._last_pilot_g_cur
            m = g_cur.shape[0]
        else:
            thetas = self.levels[-1].sample_fn(m)
            g_cur = self.grad(x_k, thetas)
            self.counter += m

        g_km2 = self.grad(x_km2, thetas)
        self.counter += m

        delta_stats = WelfordVec.zeros(self.dim)
        base_stats = WelfordVec.zeros(self.dim)
        base_stats.update_batch(g_cur)
        delta_stats.update_batch(g_cur - g_km2)

        drop_lvl = LevelState(
            x=x_k.copy(),
            x_prev=x_km2.copy(),
            cost=2,
            sample_fn=self.levels[-1].sample_fn,
            delta_stats=delta_stats,
            base_stats=base_stats,
            m_min=self.min_batch,
            delta_resamp=ResamplingAcc.zeros(self.re_part, self.dim) if self.use_resampling else None,
            base_resamp=ResamplingAcc.zeros(self.re_part, self.dim) if self.use_resampling else None,
        )
        if drop_lvl.base_resamp is not None:
            drop_lvl.base_resamp.update_batch(g_cur)
        if drop_lvl.delta_resamp is not None:
            drop_lvl.delta_resamp.update_batch(g_cur - g_km2)
        return drop_lvl

    def _check_clipping(self, opt_ml: np.ndarray, err_tol: float) -> Tuple[bool, np.ndarray]:
        if not self.policy.clip_type:
            return False, opt_ml
        if self.policy.clip_every and self.k % self.policy.clip_every != 0:
            return False, opt_ml

        if self.policy.clip_type == "full":
            if not self.finite:
                return False, opt_ml
            assert self.data_size is not None
            m_is_datasize = np.where(opt_ml == self.data_size)[0]
            if len(m_is_datasize) and int(m_is_datasize.max()) > 0:
                lvl_clip = int(m_is_datasize.max())
                ml = np.asarray([lvl.m for lvl in self.levels], dtype=float)
                cost = float(np.maximum(opt_ml - ml, 0).sum() + self.policy.aggr_cost * len(ml))
                deltas_clip = self.levels[lvl_clip:]
                opt_ml_clip = self._get_opt_ml_for_levels(deltas_clip, err_tol)
                ml_clip = np.asarray([lvl.m for lvl in deltas_clip], dtype=float)
                cost_clip = float(np.maximum(opt_ml_clip - ml_clip, 0).sum() + self.policy.aggr_cost * len(opt_ml_clip))
                if cost_clip <= cost:
                    self.levels = deltas_clip
                    self.levels[0] = self._restart_delta(self.levels[0])
                    self._g_hat = self._aggregate_recompute()
                    return True, np.asarray(opt_ml_clip, dtype=int)
            return False, opt_ml

        if self.policy.clip_type == "all":
            ml = np.asarray([lvl.m for lvl in self.levels], dtype=float)
            cost = float(np.maximum(opt_ml - ml, 0).sum() + self.policy.aggr_cost * len(ml))
            best_cost = cost
            best_i = None
            best_opt = None
            for i in range(len(self.levels)):
                deltas_clip = self.levels[i:]
                opt_ml_clip = self._get_opt_ml_for_levels(deltas_clip, err_tol)
                ml_clip = np.asarray([lvl.m for lvl in deltas_clip], dtype=float)
                cost_clip = float(np.maximum(opt_ml_clip - ml_clip, 0).sum() + self.policy.aggr_cost * len(opt_ml_clip))
                if cost_clip < best_cost:
                    best_cost = cost_clip
                    best_i = i
                    best_opt = opt_ml_clip
            if best_i is not None and best_opt is not None:
                self.levels = self.levels[best_i:]
                self.levels[0] = self._restart_delta(self.levels[0])
                self._g_hat = self._aggregate_recompute()
                return True, np.asarray(best_opt, dtype=int)
            return False, opt_ml

        return False, opt_ml
