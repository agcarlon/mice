from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np

from mice import plot_mice
from mice.core_impl import MICE
from mice.policy import DropRestartClipPolicy


@dataclass(frozen=True, slots=True)
class Traj:
    grads: np.ndarray
    iters: np.ndarray
    runtime: np.ndarray
    gap: np.ndarray


def _F(x: np.ndarray, *, sigma: float, a: float = 1.0, b: float = 100.0) -> float:
    x0, x1 = float(x[0]), float(x[1])
    return (a - x0) ** 2 + sigma**2 + b * (4.0 * sigma**4 + (x1 - x0**2) ** 2)


def _grad_f(x: np.ndarray, thetas: np.ndarray, *, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """
    Stochastic gradient of f(x, theta) as in the manuscript Rosenbrock example.

    thetas: shape (m,2), with (theta0, theta1).
    Returns gradients shape (m,2).
    """
    x0, x1 = float(x[0]), float(x[1])
    th0 = thetas[:, 0]
    th1 = thetas[:, 1]
    g0 = -2.0 * a + 4.0 * b * x0 * (x0**2 - x1 - th0**2 + th1**2) + 2.0 * x0 - 2.0 * th0
    g1 = 2.0 * b * (-x0**2 + x1 + th0**2 - th1**2)
    return np.stack([g0, g1], axis=1)


def _run_vanilla_adam(
    *,
    sigma: float,
    seed: int,
    max_grads: int,
    batch: int = 100,
    a: float = 1.0,
    b: float = 100.0,
) -> Traj:
    rng = np.random.default_rng(seed)

    x = np.array([-1.5, 2.5], dtype=float)
    x_opt = np.array([a, a**2], dtype=float)
    f_opt = _F(x_opt, sigma=sigma, a=a, b=b)

    beta1 = 0.9
    beta2 = 0.999
    adam_eps = 1e-8
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    gaps = []
    costs = []
    iters = []
    runtimes = []
    t0 = time()

    cost = 0
    k = 0
    while cost + batch <= int(max_grads):
        k += 1
        thetas = rng.normal(0.0, sigma, size=(batch, 2))
        g = _grad_f(x, thetas, a=a, b=b).mean(axis=0)
        cost += batch

        eta = 0.03 / np.sqrt(float(k))  # manuscript

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**k)
        v_hat = v / (1.0 - beta2**k)
        x = x - eta * m_hat / (np.sqrt(v_hat) + adam_eps)

        gaps.append(_F(x, sigma=sigma, a=a, b=b) - f_opt)
        costs.append(float(cost))
        iters.append(float(k))
        runtimes.append(time() - t0)

    return Traj(
        grads=np.asarray(costs, dtype=float),
        iters=np.asarray(iters, dtype=float),
        runtime=np.asarray(runtimes, dtype=float),
        gap=np.asarray(gaps, dtype=float),
    )


def _run_adam_mice(
    *,
    sigma: float,
    seed: int,
    max_grads: int,
    eps: float = 1.0,
    step: float = 0.3,
    a: float = 1.0,
    b: float = 100.0,
) -> Traj:
    rng = np.random.default_rng(seed)

    x = np.array([-1.5, 2.5], dtype=float)
    x_opt = np.array([a, a**2], dtype=float)
    f_opt = _F(x_opt, sigma=sigma, a=a, b=b)

    policy = DropRestartClipPolicy(
        drop_param=0.5,
        restart_param=0.0,
        max_hierarchy_size=100,
        aggr_cost=0.1,
        clip_type=None,
    )

    def sampler(n: int) -> np.ndarray:
        return rng.normal(0.0, sigma, size=(int(n), 2))

    dF = MICE(
        grad=lambda x_, th: _grad_f(x_, th, a=a, b=b),
        sampler=sampler,
        eps=float(eps),
        min_batch=10,
        restart_factor=10,
        max_cost=float(max_grads),
        stop_crit_norm=0.0,  # paper uses max grad budget for this example
        convex=False,
        policy=policy,
        use_resampling=True,
    )

    beta1 = 0.9
    beta2 = 0.999
    adam_eps = 1e-8
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    gaps = []
    costs = []
    iters = []
    runtimes = []
    t0 = time()

    k = 0
    while not dF.terminate:
        k += 1
        g = dF.evaluate(x)
        if dF.terminate:
            break

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**k)
        v_hat = v / (1.0 - beta2**k)
        x = x - float(step) * m_hat / (np.sqrt(v_hat) + adam_eps)

        gaps.append(_F(x, sigma=sigma, a=a, b=b) - f_opt)
        costs.append(float(dF.counter))
        iters.append(float(k))
        runtimes.append(time() - t0)

    return Traj(
        grads=np.asarray(costs, dtype=float),
        iters=np.asarray(iters, dtype=float),
        runtime=np.asarray(runtimes, dtype=float),
        gap=np.asarray(gaps, dtype=float),
    )


def _plot_two(traj_adam: Traj, traj_mice: Traj, *, title: str, outpath: Path) -> None:
    # Match legacy manuscript plotting style (old_mice/mice_numerics/rosenbrock/plots_losses.py):
    #  - 3 stacked panels
    #  - loglog for (num_grads, opt_gap) and (iteration, opt_gap)
    #  - semilogy for (runtime, opt_gap)
    import pandas as pd

    df_adam = pd.DataFrame(
        {
            "num_grads": traj_adam.grads,
            "iteration": traj_adam.iters,
            "runtime": traj_adam.runtime,
            "opt_gap": traj_adam.gap,
            "event": None,
        }
    )
    df_mice = pd.DataFrame(
        {
            "num_grads": traj_mice.grads,
            "iteration": traj_mice.iters,
            "runtime": traj_mice.runtime,
            "opt_gap": traj_mice.gap,
            "event": None,
        }
    )

    labels = ("Adam", "Adam-MICE")
    colors = ("C0", "C1")

    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    for d, c in ((df_adam, colors[0]), (df_mice, colors[1])):
        axs[0] = plot_mice(d, axs[0], x="num_grads", y="opt_gap", markers=False, color=c)
    axs[0].set_ylabel(r"$F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)$")
    axs[0].set_xlabel(r"$\#$ grads")
    axs[0].set_title(title)
    axs[0].legend(labels)

    for d, c in ((df_adam, colors[0]), (df_mice, colors[1])):
        axs[1] = plot_mice(d, axs[1], x="iteration", y="opt_gap", markers=False, color=c, style="loglog")
    axs[1].set_ylabel(r"$F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)$")
    axs[1].set_xlabel("Iteration")

    for d, c in ((df_adam, colors[0]), (df_mice, colors[1])):
        axs[2] = plot_mice(d, axs[2], x="runtime", y="opt_gap", markers=False, color=c, style="semilogy")
    axs[2].set_ylabel(r"$F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)$")
    axs[2].set_xlabel(r"Runtime (s)")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath))
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Rosenbrock example: Adam vs Adam-MICE.")
    p.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parents[1] / "output"))
    p.add_argument("--quick", action="store_true", help="Reduce runtime (not paper settings).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=1.0)
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        max_grads = 100_000
    else:
        max_grads = 10_000_000

    for sigma, outname in ((1e-4, "optimality_gap_Rosenbrock_1.pdf"), (1e-1, "optimality_gap_Rosenbrock_2.pdf")):
        print(f"Running sigma={sigma:g} (max_grads={max_grads})")
        traj_adam = _run_vanilla_adam(sigma=sigma, seed=int(args.seed), max_grads=max_grads)
        traj_mice = _run_adam_mice(sigma=sigma, seed=int(args.seed), max_grads=max_grads, eps=float(args.eps))
        outpath = outdir / outname
        sigma_exp = -4 if sigma == 1e-4 else (-1 if sigma == 1e-1 else None)
        title = (
            rf"Rosenbrock with $\sigma_{{\theta}} = 10^{{{sigma_exp}}}$"
            if sigma_exp is not None
            else f"Rosenbrock with sigma={sigma:g}"
        )
        _plot_two(
            traj_adam,
            traj_mice,
            title=title,
            outpath=outpath,
        )
        print(f"Wrote: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
