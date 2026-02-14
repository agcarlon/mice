"""
Sensitivity sweeps for δ_drop and δ_rest with All+Clip (Reviewer request).

Run SGD-MICE (All+Clip only) on the same d-dimensional quadratic with fixed gradient
budget as the operator ablation; sweep δ_drop (drop_param) with δ_rest=0 fixed, and
sweep δ_rest (restart_param) with δ_drop=0.5 fixed. Output one CSV per sweep.

Usage (from repo root):
  python -m experiments.quadratic.run_delta_sweep [options]

Output:
  - experiments/output/quadratic_delta_drop_sensitivity.csv  (one row per drop_param value)
  - experiments/output/quadratic_delta_rest_sensitivity.csv  (one row per restart_param value)
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
    drop_param: float,
    restart_param: float,
    eps: float = 0.577,
    stop_crit_norm: float = 1e-8,
    max_iter: int = 100_000,
) -> tuple[float, float, float, int, dict[str, int]]:
    """
    Run one SGD-MICE trajectory on the quadratic with All+Clip.
    drop_param = δ_drop, restart_param = δ_rest (paper notation).
    Returns (initial_gap, final_gap, relative_optimality, grad_evals_used, op_counts).
    """
    dobjf, _objf, Eobjf, _Edobjf, sampler, _optimum, f_opt, _L, stepsize = make_problem(
        kappa=kappa, seed=seed, dim=dim
    )

    # All+Clip only; vary drop_param (δ_drop) and restart_param (δ_rest) per sweep.
    policy = DropRestartClipPolicy(
        drop_param=float(drop_param),
        restart_param=float(restart_param),
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
        description="Quadratic δ_drop / δ_rest sensitivity sweeps with All+Clip (fixed budget)"
    )
    p.add_argument("--dim", type=int, default=100, help="Problem dimension")
    p.add_argument("--kappa", type=float, default=100.0, help="Condition number")
    p.add_argument("--max-cost", type=float, default=100_000, help="Max gradient evaluations per run")
    p.add_argument("--runs", type=int, default=50, help="Number of runs per parameter value")
    p.add_argument("--seed", type=int, default=2, help="Base seed (per-run seed = seed + run_index)")
    p.add_argument(
        "--sweep",
        type=str,
        choices=("drop", "rest", "both"),
        default="both",
        help="Which sweep(s) to run: drop (δ_drop), rest (δ_rest), or both (default)",
    )
    p.add_argument(
        "--drop-values",
        type=str,
        default="0,0.25,0.5,0.75,1",
        help="Comma- or space-separated δ_drop values (default: 0,0.25,0.5,0.75,1)",
    )
    p.add_argument(
        "--restart-values",
        type=str,
        default="0,0.25,0.5,0.75,1",
        help="Comma- or space-separated δ_rest values (default: 0,0.25,0.5,0.75,1)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "output"),
        help="If set, write CSVs to this directory",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Override path for drop sweep CSV (optional); rest sweep uses same dir + rest filename",
    )
    args = p.parse_args()

    n_runs = max(1, args.runs)

    def parse_floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.replace(",", " ").split() if x.strip()]

    drop_values = parse_floats(args.drop_values)
    restart_values = parse_floats(args.restart_values)

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        csv_drop = outdir / "quadratic_delta_drop_sensitivity.csv"
        csv_rest = outdir / "quadratic_delta_rest_sensitivity.csv"
    elif args.csv:
        csv_drop = Path(args.csv)
        csv_drop.parent.mkdir(parents=True, exist_ok=True)
        csv_rest = csv_drop.parent / "quadratic_delta_rest_sensitivity.csv"
    else:
        csv_drop = Path("quadratic_delta_drop_sensitivity.csv")
        csv_rest = Path("quadratic_delta_rest_sensitivity.csv")

    fieldnames_drop = [
        "drop_param", "dim", "kappa", "budget", "n_runs",
        "mean_relative_optimality", "std_relative_optimality",
        "mean_final_gap", "std_final_gap", "mean_initial_gap",
        "mean_grad_evals", "std_grad_evals",
        "mean_drop_events", "mean_restart_events", "mean_clip_events",
    ]
    fieldnames_rest = [
        "restart_param", "dim", "kappa", "budget", "n_runs",
        "mean_relative_optimality", "std_relative_optimality",
        "mean_final_gap", "std_final_gap", "mean_initial_gap",
        "mean_grad_evals", "std_grad_evals",
        "mean_drop_events", "mean_restart_events", "mean_clip_events",
    ]

    # Sweep 1: δ_drop (restart_param = 0)
    if args.sweep in ("drop", "both"):
        rows_drop = []
        for drop_param in drop_values:
            runs_data = []
            print(
                f"\nSweep δ_drop: drop_param={drop_param} (restart_param=0, dim={args.dim}, "
                f"kappa={args.kappa}, budget={args.max_cost}, runs={n_runs})"
            )
            for r in range(n_runs):
                seed = args.seed + r
                gap_0, gap_T, rel_optimality, grad_evals, op_counts = run_single(
                    dim=args.dim,
                    kappa=args.kappa,
                    seed=seed,
                    max_cost=args.max_cost,
                    drop_param=drop_param,
                    restart_param=0.0,
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
                "drop_param": drop_param,
                "dim": args.dim,
                "kappa": args.kappa,
                "budget": args.max_cost,
                "n_runs": n_runs,
                **agg,
            }
            rows_drop.append(row)
            print(
                f"  Summary: mean_relative_optimality = {agg['mean_relative_optimality']:.4f} ± {agg['std_relative_optimality']:.4f}, "
                f"mean events: drop={agg['mean_drop_events']:.1f}, restart={agg['mean_restart_events']:.1f}, clip={agg['mean_clip_events']:.1f}"
            )
        with open(csv_drop, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames_drop)
            writer.writeheader()
            writer.writerows(rows_drop)
        print(f"\nWrote {csv_drop}")

    # Sweep 2: δ_rest (drop_param = 0.5)
    if args.sweep in ("rest", "both"):
        rows_rest = []
        for restart_param in restart_values:
            runs_data = []
            print(
                f"\nSweep δ_rest: restart_param={restart_param} (drop_param=0.5, dim={args.dim}, "
                f"kappa={args.kappa}, budget={args.max_cost}, runs={n_runs})"
            )
            for r in range(n_runs):
                seed = args.seed + r
                gap_0, gap_T, rel_optimality, grad_evals, op_counts = run_single(
                    dim=args.dim,
                    kappa=args.kappa,
                    seed=seed,
                    max_cost=args.max_cost,
                    drop_param=0.5,
                    restart_param=restart_param,
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
                "restart_param": restart_param,
                "dim": args.dim,
                "kappa": args.kappa,
                "budget": args.max_cost,
                "n_runs": n_runs,
                **agg,
            }
            rows_rest.append(row)
            print(
                f"  Summary: mean_relative_optimality = {agg['mean_relative_optimality']:.4f} ± {agg['std_relative_optimality']:.4f}, "
                f"mean events: drop={agg['mean_drop_events']:.1f}, restart={agg['mean_restart_events']:.1f}, clip={agg['mean_clip_events']:.1f}"
            )
        with open(csv_rest, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames_rest)
            writer.writeheader()
            writer.writerows(rows_rest)
        print(f"\nWrote {csv_rest}")


if __name__ == "__main__":
    main()
