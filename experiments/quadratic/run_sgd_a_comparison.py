from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np

from mice.core_impl import MICE
from mice.policy import DropRestartClipPolicy

from .problem import make_problem


def _run_sgd_mice(
    *,
    kappa: float,
    seed: int,
    eps: float,
    tol: float,
    max_cost: float,
    max_iter: int,
) -> dict[str, np.ndarray]:
    dobjf, _objf, Eobjf, _Edobjf, sampler, _optimum, f_opt, L, stepsize = make_problem(
        kappa=kappa, seed=seed, dim=2
    )

    policy = DropRestartClipPolicy(
        drop_param=0.5,
        restart_param=0.0,
        max_hierarchy_size=100,
        aggr_cost=0.1,
        clip_type="all",
    )

    dF = MICE(
        grad=dobjf,
        sampler=sampler,
        eps=float(eps),
        min_batch=10,
        restart_factor=10,
        stop_crit_norm=float(tol),
        stop_crit_prob=0.05,
        convex=False,
        use_resampling=True,
        max_cost=float(max_cost),
        policy=policy,
    )

    x = np.array([20.0, 50.0], dtype=float)
    gaps = []
    costs = []
    iters = []
    runtimes = []
    t0 = time()
    for k in range(int(max_iter)):
        g = dF.evaluate(x)
        if dF.terminate:
            break
        x = x - stepsize * g
        gaps.append(Eobjf(x) - f_opt)
        costs.append(float(dF.counter))
        iters.append(float(k))
        runtimes.append(time() - t0)
    return {
        "gap": np.asarray(gaps, dtype=float),
        "cost": np.asarray(costs, dtype=float),
        "iter": np.asarray(iters, dtype=float),
        "runtime": np.asarray(runtimes, dtype=float),
        "final_cost": np.asarray([float(dF.counter)], dtype=float),
    }


def _run_sgd_a(
    *,
    kappa: float,
    seed: int,
    eps: float,
    tol: float,
    max_cost: float,
    max_iter: int,
) -> dict[str, np.ndarray]:
    """
    SGD-A: SGD with adaptive Monte Carlo mean-gradient sampling.

    As stated in the manuscript, this is a special case of SGD-MICE where Restart is used every iteration.
    We implement it by forcing MICE to restart at each evaluate() call, so the hierarchy stays length 1.
    """
    dobjf, _objf, Eobjf, _Edobjf, sampler, _optimum, f_opt, L, stepsize = make_problem(
        kappa=kappa, seed=seed, dim=2
    )

    policy = DropRestartClipPolicy(
        drop_param=0.0,
        restart_param=0.0,
        max_hierarchy_size=1,
        aggr_cost=0.0,
        clip_type=None,
    )

    dF = MICE(
        grad=dobjf,
        sampler=sampler,
        eps=float(eps),
        min_batch=10,
        restart_factor=10,
        stop_crit_norm=float(tol),
        stop_crit_prob=0.05,
        convex=False,
        use_resampling=True,
        max_cost=float(max_cost),
        policy=policy,
    )

    x = np.array([20.0, 50.0], dtype=float)
    gaps = []
    costs = []
    iters = []
    runtimes = []
    t0 = time()
    for k in range(int(max_iter)):
        dF.force_restart = True
        g = dF.evaluate(x)
        if dF.terminate:
            break
        x = x - stepsize * g
        gaps.append(Eobjf(x) - f_opt)
        costs.append(float(dF.counter))
        iters.append(float(k))
        runtimes.append(time() - t0)
    return {
        "gap": np.asarray(gaps, dtype=float),
        "cost": np.asarray(costs, dtype=float),
        "iter": np.asarray(iters, dtype=float),
        "runtime": np.asarray(runtimes, dtype=float),
        "final_cost": np.asarray([float(dF.counter)], dtype=float),
    }


def _run_vanilla_sgd(
    *,
    kappa: float,
    seed: int,
    batch_size: int,
    max_cost: float,
    max_iter: int,
) -> dict[str, np.ndarray]:
    dobjf, _objf, Eobjf, _Edobjf, sampler, _optimum, f_opt, L, _stepsize = make_problem(
        kappa=kappa, seed=seed, dim=2
    )

    rng = np.random.default_rng(seed)
    x = np.array([20.0, 50.0], dtype=float)
    gaps = []
    costs = []
    iters = []
    runtimes = []
    t0 = time()
    cost = 0
    for k in range(int(max_iter)):
        if cost + batch_size > max_cost:
            break
        thetas = rng.uniform(0.0, 1.0, int(batch_size))
        g = dobjf(x, thetas).mean(axis=0)
        # Manuscript notes a decaying stepsize for vanilla SGD tuned for performance:
        #   eta_k = (L(1 + k/50))^{-1}
        eta = 1.0 / (float(L) * (1.0 + float(k) / 50.0))
        x = x - eta * g
        cost += batch_size
        gaps.append(Eobjf(x) - f_opt)
        costs.append(float(cost))
        iters.append(float(k))
        runtimes.append(time() - t0)
    return {
        "gap": np.asarray(gaps, dtype=float),
        "cost": np.asarray(costs, dtype=float),
        "iter": np.asarray(iters, dtype=float),
        "runtime": np.asarray(runtimes, dtype=float),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Quadratic comparison: SGD-A vs SGD-MICE vs vanilla SGD.")
    p.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parents[1] / "output"))
    p.add_argument("--quick", action="store_true", help="Reduce runtime (not paper settings).")
    p.add_argument("--kappa", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eps", type=float, default=float(np.sqrt(1.0 / 3.0)))
    p.add_argument("--tol", type=float, default=1e-8, help="Stopping criterion on ||∇F||^2.")
    p.add_argument("--max-iter", type=int, default=2_500_000)
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        tol = 1e-4
        max_cost = 50_000.0
        max_iter = 20_000
        cap_iter_plot = 500
    else:
        tol = float(args.tol)
        max_cost = float("inf")
        max_iter = int(args.max_iter)
        cap_iter_plot = 1400

    sgd_mice = _run_sgd_mice(
        kappa=float(args.kappa),
        seed=int(args.seed),
        eps=float(args.eps),
        tol=tol,
        max_cost=max_cost,
        max_iter=max_iter,
    )
    sgd_a = _run_sgd_a(
        kappa=float(args.kappa),
        seed=int(args.seed),
        eps=float(args.eps),
        tol=tol,
        max_cost=max_cost,
        max_iter=max_iter,
    )

    # Run vanilla SGD up to the same work as SGD-A (as in the manuscript).
    max_cost_vanilla = float(sgd_a["final_cost"][0]) if len(sgd_a["final_cost"]) else max_cost
    vanilla = _run_vanilla_sgd(
        kappa=float(args.kappa),
        seed=int(args.seed) + 123,  # separate RNG stream
        batch_size=1000,
        max_cost=max_cost_vanilla,
        max_iter=max_iter,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Top: optimality gap vs iteration (cap the x-range for readability as in manuscript).
    def _plot_gap_vs_iter(ax, data, label, color):
        if len(data["iter"]) == 0:
            return
        mask = data["iter"] <= cap_iter_plot
        ax.semilogy(data["iter"][mask], data["gap"][mask], label=label, color=color)

    _plot_gap_vs_iter(ax1, sgd_a, "SGD-A", "C1")
    _plot_gap_vs_iter(ax1, sgd_mice, "SGD-MICE", "C0")
    _plot_gap_vs_iter(ax1, vanilla, "SGD (batch=1000)", "C2")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"$F(\xi_k) - F(\xi^*)$")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.2)

    # Bottom: optimality gap vs gradient cost
    def _plot_gap_vs_cost(ax, data, label, color):
        if len(data["cost"]) == 0:
            return
        ax.semilogy(data["cost"], data["gap"], label=label, color=color)

    _plot_gap_vs_cost(ax2, sgd_a, "SGD-A", "C1")
    _plot_gap_vs_cost(ax2, sgd_mice, "SGD-MICE", "C0")
    _plot_gap_vs_cost(ax2, vanilla, "SGD (batch=1000)", "C2")
    ax2.set_xlabel(r"Gradient sampling cost $\mathcal{C}_k$")
    ax2.set_ylabel(r"$F(\xi_k) - F(\xi^*)$")
    ax2.grid(True, which="both", alpha=0.2)

    # Show tol as a horizontal reference on the gap scale? Manuscript uses tol on gradient norm,
    # so we only annotate the stop criterion numerically.
    fig.suptitle(f"Quadratic example (kappa={args.kappa:g})")
    fig.tight_layout()

    outpath = outdir / "sgd_a.pdf"
    fig.savefig(str(outpath))
    plt.close(fig)
    print(f"Wrote: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

