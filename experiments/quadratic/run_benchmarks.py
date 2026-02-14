"""
Run MICE with TimedMICE and print overhead breakdown.

Referee request: "micro-benchmarks of MICE's aggregation/variance-estimation
overhead" and "ablation isolating the cost of (i) variance estimates,
(ii) resampling for stopping, and (iii) index-set maintenance".

Usage (from repo root):
  python -m experiments.quadratic.run_benchmarks [options]

Or directly:
  python experiments/quadratic/run_benchmarks.py [options]

Default outputs are written to:
  experiments/output/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as script or as module
if __name__ == "__main__" and __file__ is not None:
    _bench_dir = Path(__file__).resolve().parent
    if str(_bench_dir.parent) not in sys.path:
        sys.path.insert(0, str(_bench_dir.parent.parent.parent))
else:
    _bench_dir = Path(__file__).resolve().parent

from .timed_mice import TimedMICE


def run_quadratic(
    *,
    kappa: float = 100.0,
    seed: int = 1,
    eps: float = 0.577,
    use_resampling: bool = True,
    max_cost: float = 50_000,
    stop_crit_norm: float = 1e-6,
    max_iter: int = 5000,
    dim: int = 2,
):
    """Run quadratic problem with TimedMICE; return (log, timing dict, dF)."""
    from .problem import make_problem

    dobjf, _objf, Eobjf, Edobjf, sampler, optimum, f_opt, L, stepsize = make_problem(
        kappa=kappa, seed=seed, dim=dim
    )

    dF = TimedMICE(
        grad=dobjf,
        sampler=sampler,
        eps=float(eps),
        min_batch=10,
        restart_factor=10,
        stop_crit_norm=float(stop_crit_norm),
        stop_crit_prob=0.05,
        convex=False,
        use_resampling=bool(use_resampling),
        max_cost=float(max_cost),
    )

    # Start from a point away from the optimum; scale with dimension.
    x0 = np.full(dim, 20.0, dtype=float)
    X = [x0]
    while (not dF.terminate) and (len(X) < int(max_iter)):
        g = dF.evaluate(X[-1])
        if dF.terminate:
            break
        X.append(X[-1] - stepsize * g)

    log = dF.get_log()
    timing = dict(dF.timing)
    return log, timing, dF


# Keys that define total runtime (exclude index_set sub-keys to avoid double-count)
_MAIN_TIMING_KEYS = ("gradient", "variance_sizing", "resampling", "index_set")


def print_breakdown(timing: dict, title: str = "Overhead breakdown") -> None:
    total = sum(timing.get(k, 0.0) for k in _MAIN_TIMING_KEYS)
    print(f"\n{title}")
    print("-" * 56)
    for key in _MAIN_TIMING_KEYS:
        t = timing.get(key, 0.0)
        pct = (100.0 * t / total) if total > 0 else 0.0
        print(f"  {key:20s}  {t:8.4f} s   ({pct:5.1f}%)")
    print("-" * 56)
    print(f"  {'total':20s}  {total:8.4f} s")
    overhead = total - timing.get("gradient", 0.0)
    if total > 0:
        print(f"  (overhead = total - gradient = {overhead:.4f} s, {100.0 * overhead / total:.1f}%)")


def main():
    p = argparse.ArgumentParser(description="MICE overhead micro-benchmark")
    p.add_argument("--kappa", type=float, default=100.0, help="Condition number (quadratic)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eps", type=float, default=0.577, help="Relative error tolerance")
    p.add_argument("--dim", type=int, default=2, help="Problem dimension for single run")
    p.add_argument(
        "--dims",
        type=str,
        nargs="+",
        default=None,
        help="Dimension list for sweep mode (examples: '--dims 10,100,1000' or '--dims 10, 100, 1000')",
    )
    p.add_argument("--no-resampling", action="store_true", help="Disable resampling (plain norm)")
    p.add_argument("--max-cost", type=float, default=100_000, help="Cap on gradient evaluations")
    p.add_argument("--stop-crit", type=float, default=0., help="Stopping ||∇F||^2 tolerance")
    p.add_argument("--max-iter", type=int, default=100_000)
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per dimension in sweep mode; percentages are averaged (default: 1)",
    )
    default_outdir = Path(__file__).resolve().parents[1] / "output"
    p.add_argument("--outdir", type=str, default=str(default_outdir), help="Directory for CSV/PDF outputs")
    args = p.parse_args()
    # Sweep mode: run across multiple dimensions and aggregate percentages.
    if args.dims:
        # Parse list of dimensions (comma- or space-separated)
        dims_raw = " ".join(args.dims)
        tokens = dims_raw.replace(",", " ").split()
        dims = sorted({int(tok) for tok in tokens})
        n_runs = max(1, int(args.runs))

        results = []
        for d in dims:
            run_pcts = []
            run_idx_pcts = []
            run_totals = []
            for r in range(n_runs):
                seed = args.seed + r
                log, timing, dF = run_quadratic(
                    kappa=args.kappa,
                    seed=seed,
                    eps=args.eps,
                    use_resampling=not args.no_resampling,
                    max_cost=args.max_cost,
                    stop_crit_norm=args.stop_crit,
                    max_iter=args.max_iter,
                    dim=d,
                )
                total = sum(timing.get(k, 0.0) for k in _MAIN_TIMING_KEYS)
                if total <= 0.0:
                    pct = {k: 0.0 for k in _MAIN_TIMING_KEYS}
                else:
                    pct = {
                        k: 100.0 * float(timing.get(k, 0.0)) / float(total)
                        for k in _MAIN_TIMING_KEYS
                    }
                idx_total = sum(
                    timing.get(k, 0.0)
                    for k in (
                        "index_set_add",
                        "index_set_drop",
                        "index_set_restart",
                        "index_set_clip",
                    )
                )
                if idx_total <= 0.0:
                    idx_pct = {"add": 0.0, "drop": 0.0, "restart": 0.0, "clip": 0.0}
                else:
                    idx_pct = {
                        "add": 100.0 * timing.get("index_set_add", 0.0) / idx_total,
                        "drop": 100.0 * timing.get("index_set_drop", 0.0) / idx_total,
                        "restart": 100.0 * timing.get("index_set_restart", 0.0) / idx_total,
                        "clip": 100.0 * timing.get("index_set_clip", 0.0) / idx_total,
                    }
                run_pcts.append(pct)
                run_idx_pcts.append(idx_pct)
                run_totals.append(float(total))
                if n_runs > 1:
                    print(
                        f"  dim={d} run {r + 1}/{n_runs}: total={total:.3f}s, "
                        f"grad_evals={int(dF.counter)}"
                    )

            # Average percentages and total time across runs
            avg_pct = {
                k: float(np.mean([p[k] for p in run_pcts]))
                for k in _MAIN_TIMING_KEYS
            }
            avg_idx_pct = {
                name: float(np.mean([p[name] for p in run_idx_pcts]))
                for name in ("add", "drop", "restart", "clip")
            }
            avg_total = float(np.mean(run_totals))
            results.append(
                {
                    "dim": d,
                    "pct_gradient": avg_pct["gradient"],
                    "pct_variance_sizing": avg_pct["variance_sizing"],
                    "pct_resampling": avg_pct["resampling"],
                    "pct_index_set": avg_pct["index_set"],
                    "total_time": avg_total,
                    "pct_idx_add": avg_idx_pct["add"],
                    "pct_idx_drop": avg_idx_pct["drop"],
                    "pct_idx_restart": avg_idx_pct["restart"],
                    "pct_idx_clip": avg_idx_pct["clip"],
                    "n_runs": n_runs,
                }
            )
            title = f"MICE overhead (quadratic, κ={args.kappa}, dim={d})"
            if n_runs > 1:
                title += f" — avg of {n_runs} runs"
            print_breakdown(
                {k: avg_total * avg_pct[k] / 100.0 for k in _MAIN_TIMING_KEYS},
                title=title,
            )
            print(f"  (averaged total time: {avg_total:.4f} s)")

        # Save CSV and stacked percentage plot if requested
        outdir = Path(args.outdir) if args.outdir else Path(".")
        outdir.mkdir(parents=True, exist_ok=True)

        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(results).sort_values("dim")
        df.to_csv(outdir / "overhead_by_dim.csv", index=False)

        dims_arr = df["dim"].to_numpy(dtype=float)
        p_grad = df["pct_gradient"].to_numpy(dtype=float)
        p_var = df["pct_variance_sizing"].to_numpy(dtype=float)
        p_res = df["pct_resampling"].to_numpy(dtype=float)
        p_idx = df["pct_index_set"].to_numpy(dtype=float)

        # Cumulative bands: 0 -> grad -> grad+var -> grad+var+res -> 100
        c0 = np.zeros_like(dims_arr)
        c1 = p_grad
        c2 = p_grad + p_var
        c3 = c2 + p_res
        c4 = np.full_like(dims_arr, 100.0)

        # Index-set operator shares (of index_set time only), stacked 0-100%
        p_add = df["pct_idx_add"].to_numpy(dtype=float)
        p_drop = df["pct_idx_drop"].to_numpy(dtype=float)
        p_restart = df["pct_idx_restart"].to_numpy(dtype=float)
        p_clip = df["pct_idx_clip"].to_numpy(dtype=float)
        ic0 = np.zeros_like(dims_arr)
        ic1 = p_add
        ic2 = p_add + p_drop
        ic3 = ic2 + p_restart
        ic4 = np.full_like(dims_arr, 100.0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
        # Top: total runtime breakdown (same style as before)
        ax1.set_title("Runtime contribution - Quadratic problem")
        ax1.fill_between(dims_arr, c0, c1, alpha=0.5, label="gradient")
        ax1.fill_between(dims_arr, c1, c2, alpha=0.5, label="variance estimation")
        ax1.fill_between(dims_arr, c2, c3, alpha=0.5, label="resampling")
        ax1.fill_between(dims_arr, c3, c4, alpha=0.5, label="index set")
        ax1.set_xlim(dims_arr.min(), dims_arr.max())
        ax1.set_ylabel(r"Runtime share ($\%$)")
        ax1.set_ylim(0.0, 100.0)
        ax1.set_xscale("log")
        ax1.legend(loc="lower right")

        # Bottom: index-set operator breakdown (same style, 0-100% of index_set time)
        ax2.set_title("Index-set operator share (of index-set time)")
        ax2.fill_between(dims_arr, ic0, ic1, alpha=0.5, label="Add")
        ax2.fill_between(dims_arr, ic1, ic2, alpha=0.5, label="Drop")
        ax2.fill_between(dims_arr, ic2, ic3, alpha=0.5, label="Restart")
        ax2.fill_between(dims_arr, ic3, ic4, alpha=0.5, label="Clip")
        ax2.set_xlabel("Dimension")
        ax2.set_ylabel(r"Share of index-set time ($\%$)")
        ax2.set_ylim(0.0, 100.0)
        ax2.set_xscale("log")
        ax2.legend(loc="lower right")

        fig.tight_layout()
        fig.savefig(outdir / "overhead_pct_vs_dim.pdf")
        plt.close(fig)

        print(f"\nSaved sweep results to {outdir/'overhead_by_dim.csv'}")
        print(f"Saved stacked percentage plot (runtime + index-set ops) to {outdir/'overhead_pct_vs_dim.pdf'}")
        return

    # Single-dimension run (backward-compatible behavior)
    log, timing, dF = run_quadratic(
        kappa=args.kappa,
        seed=args.seed,
        eps=args.eps,
        use_resampling=not args.no_resampling,
        max_cost=args.max_cost,
        stop_crit_norm=args.stop_crit,
        max_iter=args.max_iter,
        dim=args.dim,
    )

    print_breakdown(
        timing,
        title="MICE overhead (quadratic, κ=%s, dim=%d)" % (args.kappa, args.dim),
    )
    print(
        f"\nRun ended: terminate={dF.terminate}, reason={dF.terminate_reason}, "
        f"iterations={len(log)}, grad_evals={int(dF.counter)}"
    )

    if args.outdir:
        import pandas as pd
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        df = log if hasattr(log, "to_pickle") else pd.DataFrame(log)
        df.to_pickle(str(out / "overhead_bench_log.pkl"))
        with open(out / "overhead_bench_timing.txt", "w") as f:
            for k, v in timing.items():
                f.write(f"{k}\t{v}\n")
        print(f"Saved log and timing to {out}")


if __name__ == "__main__":
    main()
