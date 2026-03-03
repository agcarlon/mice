from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from mice.core_impl import MICE
from mice.policy import DropRestartClipPolicy

from .problem import make_problem


@dataclass(frozen=True, slots=True)
class ConsistencyResult:
    tol: float
    norms2: np.ndarray  # true ||∇F||^2 at stop
    pct_above: float


def _run_many(
    *,
    kappa: float,
    eps: float,
    tol: float,
    n_runs: int,
    seed: int,
    use_resampling: bool,
    max_cost: float,
    max_iter: int,
) -> ConsistencyResult:
    dobjf, _objf, _Eobjf, Edobjf, sampler, _optimum, _f_opt, _L, stepsize = make_problem(
        kappa=kappa, seed=seed, dim=2
    )

    policy = DropRestartClipPolicy(
        drop_param=0.5,
        restart_param=0.0,
        max_hierarchy_size=100,
        aggr_cost=0.1,
        clip_type="all",
    )

    norms2 = np.zeros(int(n_runs), dtype=float)
    for r in range(int(n_runs)):
        # Different RNG stream per run
        _seed = seed + r
        np.random.seed(_seed)

        dF = MICE(
            grad=dobjf,
            sampler=sampler,
            eps=float(eps),
            min_batch=10,
            restart_factor=10,
            stop_crit_norm=float(tol),
            stop_crit_prob=0.05,
            convex=False,
            use_resampling=bool(use_resampling),
            max_cost=float(max_cost),
            policy=policy,
        )

        x = np.array([20.0, 50.0], dtype=float)
        for _k in range(int(max_iter)):
            g = dF.evaluate(x)
            if dF.terminate:
                break
            x = x - stepsize * g

        tg = np.asarray(Edobjf(x), dtype=float)
        norms2[r] = float(tg @ tg)

    pct_above = 100.0 * float(np.mean(norms2 > float(tol)))
    return ConsistencyResult(tol=float(tol), norms2=norms2, pct_above=pct_above)


def _violin(ax, datasets: list[ConsistencyResult], *, title: str) -> None:
    # Build violins in log10 space for stability across tolerances.
    positions = np.arange(len(datasets), dtype=float)

    for i, res in enumerate(datasets):
        y = np.asarray(res.norms2, dtype=float)
        y = y[np.isfinite(y) & (y > 0)]
        if y.size == 0:
            continue
        ylog = np.log10(y)

        kde = gaussian_kde(ylog)
        grid = np.linspace(ylog.min(), ylog.max(), 200)
        dens = kde(grid)
        dens = dens / (dens.max() + 1e-12)
        width = 0.35 * dens

        x0 = positions[i]
        ygrid = 10 ** grid
        ax.fill_betweenx(ygrid, x0 - width, x0 + width, color="#8ecae6", alpha=0.6, linewidth=0)

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        q25, q50, q75 = np.quantile(y, [0.25, 0.5, 0.75])

        # Thin hair: min/max
        ax.plot([x0, x0], [y_min, y_max], color="k", lw=0.8)
        # Thick hair: IQR
        ax.plot([x0, x0], [q25, q75], color="k", lw=3.0)
        # Median dot
        ax.plot([x0], [q50], "wo", ms=5, mec="k", mew=0.6)

        # Tol line and % above tol
        ax.hlines(res.tol, x0 - 0.45, x0 + 0.45, colors="k", linestyles="--", lw=1.0, alpha=0.8)
        ax.text(x0, res.tol * 1.5, f"{res.pct_above:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_yscale("log")
    ax.set_xlim(-0.8, len(datasets) - 0.2)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{r.tol:.0e}" for r in datasets])
    ax.set_xlabel(r"$tol$")
    ax.set_ylabel(r"$\|\nabla F(\xi_{k^*})\|^2$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)


def main() -> int:
    p = argparse.ArgumentParser(description="Quadratic stopping consistency violin plots.")
    p.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parents[1] / "output"))
    p.add_argument("--quick", action="store_true", help="Reduce runtime (not paper settings).")
    p.add_argument("--kappa", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=float(np.sqrt(1.0 / 3.0)))
    p.add_argument(
        "--tols",
        type=str,
        default="1e-2, 1e-4, 1e-6, 1e-8",
        help="Comma- or space-separated tolerances (interpreted as ||∇F||^2 thresholds).",
    )
    p.add_argument("--runs", type=int, default=1000, help="Number of independent runs per tol.")
    p.add_argument("--max-cost", type=float, default=float("inf"), help="Optional cap on gradient evaluations per run.")
    p.add_argument("--max-iter", type=int, default=int(1e7), help="Hard cap on iterations per run.")
    args = p.parse_args()

    def _parse_floats(s: str) -> list[float]:
        toks = [t for t in s.replace(",", " ").split() if t.strip()]
        return [float(t) for t in toks]

    tols = _parse_floats(args.tols)
    if args.quick:
        tols = tols[:3]
        n_runs = min(int(args.runs), 50)
        max_cost = 100_000.0 if np.isfinite(args.max_cost) else 100_000.0
        max_iter = 50_000
    else:
        n_runs = int(args.runs)
        max_cost = float(args.max_cost)
        max_iter = int(args.max_iter)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for use_resampling, outname, title in (
        (True, "consistency_plot_violin.pdf", "Consistency with resampling"),
        (False, "consistency_plot_violinno_resamp.pdf", "Consistency without resampling"),
    ):
        results = []
        print(f"\nRunning: {outname} (runs={n_runs}, tols={tols})")
        for t in tols:
            res = _run_many(
                kappa=float(args.kappa),
                eps=float(args.eps),
                tol=float(t),
                n_runs=n_runs,
                seed=int(args.seed) + int(10_000 * t) + (0 if use_resampling else 1_000_000),
                use_resampling=use_resampling,
                max_cost=max_cost,
                max_iter=max_iter,
            )
            results.append(res)
            print(f"  tol={t:.0e} pct_above={res.pct_above:.2f}%")

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        _violin(ax, results, title=title)
        fig.tight_layout()
        outpath = outdir / outname
        fig.savefig(str(outpath))
        plt.close(fig)
        print(f"Wrote: {outpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

