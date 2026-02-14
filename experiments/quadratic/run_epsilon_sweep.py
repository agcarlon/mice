"""
Sensitivity sweep for the relative-error tolerance ε (Reviewer request).

Run SGD-MICE (All+Clip) on the same d-dimensional quadratic with fixed gradient
budget; sweep ε (eps) with standard δ_drop=0.5, δ_rest=0. Output one CSV.

Usage (from repo root):
  python -m experiments.quadratic.run_epsilon_sweep [options]

Output:
  - experiments/output/quadratic_epsilon_sensitivity.csv  (one row per eps value)
"""

from __future__ import annotations

import argparse
import csv as csv_module
import sys
from pathlib import Path

import numpy as np

# Ensure package is importable when run as script (e.g. debugger running the file directly)
if __name__ == "__main__" and __file__ is not None:
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from mice.core_impl import MICE
from mice.policy import DropRestartClipPolicy

if __package__ is None or __package__ == "":
    from experiments.quadratic.problem import make_problem
else:
    from .problem import make_problem


def run_single(
    *,
    dim: int,
    kappa: float,
    seed: int,
    max_cost: float,
    eps: float,
    stop_crit_norm: float = 1e-8,
    max_iter: int = 100_000,
) -> tuple[float, float, float, int, dict[str, int]]:
    """
    Run one SGD-MICE trajectory on the quadratic with All+Clip.
    eps = ε (relative-error tolerance, paper notation).
    Returns (initial_gap, final_gap, relative_optimality, grad_evals_used, op_counts).
    """
    dobjf, _objf, Eobjf, _Edobjf, sampler, _optimum, f_opt, _L, stepsize = make_problem(
        kappa=kappa, seed=seed, dim=dim
    )

    # All+Clip with standard δ_drop=0.5, δ_rest=0; vary eps (ε) per sweep.
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
        min_batch=5,
        restart_factor=10,
        stop_crit_norm=float(stop_crit_norm),
        stop_crit_prob=0.05,
        convex=False,
        use_resampling=True,
        max_cost=float(max_cost),
        policy=policy,
    )

    x0 = np.full(dim, 20.0, dtype=float)
    X = [x0]
    while (not dF.terminate) and (len(X) < max_iter):
        g = dF.evaluate(X[-1])
        if dF.terminate:
            break
        X.append(X[-1] - stepsize * g)

    op_counts = {"dropped": 0, "restart": 0, "clip": 0}
    for row in dF.get_log():
        ev = row.get("event")
        if ev in op_counts:
            op_counts[ev] += 1

    gap_0 = float(Eobjf(x0) - f_opt)
    gap_T = float(Eobjf(X[-1]) - f_opt)
    rel_optimality = float(gap_T / gap_0) if gap_0 > 0 else 0.0
    grad_evals = int(dF.counter)
    return gap_0, gap_T, rel_optimality, grad_evals, op_counts


def _aggregate_runs(
    runs_data: list[dict],
    n_runs: int,
) -> dict:
    """Compute mean/std summary from runs_data (same keys as ablation)."""
    rel_optimalitys = np.array([d["relative_optimality"] for d in runs_data], dtype=float)
    final_gaps = np.array([d["final_gap"] for d in runs_data], dtype=float)
    initial_gaps = np.array([d["initial_gap"] for d in runs_data], dtype=float)
    grad_evals_arr = np.array([d["grad_evals"] for d in runs_data], dtype=float)
    drop_events_arr = np.array([d["num_drop_events"] for d in runs_data], dtype=float)
    restart_events_arr = np.array([d["num_restart_events"] for d in runs_data], dtype=float)
    clip_events_arr = np.array([d["num_clip_events"] for d in runs_data], dtype=float)

    return {
        "mean_relative_optimality": float(np.mean(rel_optimalitys)),
        "std_relative_optimality": float(np.std(rel_optimalitys, ddof=1)) if n_runs > 1 else 0.0,
        "mean_final_gap": float(np.mean(final_gaps)),
        "std_final_gap": float(np.std(final_gaps, ddof=1)) if n_runs > 1 else 0.0,
        "mean_initial_gap": float(np.mean(initial_gaps)),
        "mean_grad_evals": float(np.mean(grad_evals_arr)),
        "std_grad_evals": float(np.std(grad_evals_arr, ddof=1)) if n_runs > 1 else 0.0,
        "mean_drop_events": float(np.mean(drop_events_arr)),
        "mean_restart_events": float(np.mean(restart_events_arr)),
        "mean_clip_events": float(np.mean(clip_events_arr)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Quadratic ε (eps) sensitivity sweep with All+Clip (fixed budget)"
    )
    p.add_argument("--dim", type=int, default=100, help="Problem dimension")
    p.add_argument("--kappa", type=float, default=100.0, help="Condition number")
    p.add_argument("--max-cost", type=float, default=100_000, help="Max gradient evaluations per run")
    p.add_argument("--runs", type=int, default=50, help="Number of runs per epsilon value")
    p.add_argument("--seed", type=int, default=1, help="Base seed (per-run seed = seed + run_index)")
    p.add_argument(
        "--eps-values",
        type=str,
        default="0.3,0.5,0.577,0.7,1",
        help="Comma- or space-separated ε values (default: 0.3,0.5,0.577,0.7,1)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "output"),
        help="If set, write CSV to this directory",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV path (default: <outdir>/quadratic_epsilon_sensitivity.csv)",
    )
    args = p.parse_args()

    n_runs = max(1, args.runs)

    def parse_floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.replace(",", " ").split() if x.strip()]

    eps_values = parse_floats(args.eps_values)
    if not eps_values:
        raise ValueError("No epsilon values. Use --eps-values to specify at least one value.")

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / "quadratic_epsilon_sensitivity.csv"
    elif args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        csv_path = Path("quadratic_epsilon_sensitivity.csv")

    fieldnames = [
        "eps", "dim", "kappa", "budget", "n_runs",
        "mean_relative_optimality", "std_relative_optimality",
        "mean_final_gap", "std_final_gap", "mean_initial_gap",
        "mean_grad_evals", "std_grad_evals",
        "mean_drop_events", "mean_restart_events", "mean_clip_events",
    ]

    rows = []
    for eps in eps_values:
        runs_data = []
        print(
            f"\nSweep ε: eps={eps} (dim={args.dim}, kappa={args.kappa}, "
            f"budget={args.max_cost}, runs={n_runs})"
        )
        for r in range(n_runs):
            seed = args.seed + r
            gap_0, gap_T, rel_optimality, grad_evals, op_counts = run_single(
                dim=args.dim,
                kappa=args.kappa,
                seed=seed,
                max_cost=args.max_cost,
                eps=eps,
                stop_crit_norm=1e-8,
                max_iter=100_000,
            )
            runs_data.append({
                "initial_gap": gap_0,
                "final_gap": gap_T,
                "relative_optimality": rel_optimality,
                "grad_evals": grad_evals,
                "num_drop_events": int(op_counts.get("dropped", 0)),
                "num_restart_events": int(op_counts.get("restart", 0)),
                "num_clip_events": int(op_counts.get("clip", 0)),
            })
            print(
                f"  run {r + 1}/{n_runs}: rel_optimality={rel_optimality:.4f}, "
                f"final_gap={gap_T:.2e}, grads={grad_evals}, "
                f"drop={op_counts.get('dropped', 0)}, restart={op_counts.get('restart', 0)}, clip={op_counts.get('clip', 0)}"
            )
        agg = _aggregate_runs(runs_data, n_runs)
        row = {
            "eps": eps,
            "dim": args.dim,
            "kappa": args.kappa,
            "budget": args.max_cost,
            "n_runs": n_runs,
            **agg,
        }
        rows.append(row)
        print(
            f"  Summary: mean_relative_optimality = {agg['mean_relative_optimality']:.4f} ± {agg['std_relative_optimality']:.4f}, "
            f"mean events: drop={agg['mean_drop_events']:.1f}, restart={agg['mean_restart_events']:.1f}, clip={agg['mean_clip_events']:.1f}"
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
