"""
Run SGD-MICE on the d-dimensional quadratic with a fixed gradient budget,
repeat n times, and record the average relative optimality gap decrease for a paper table.

This script is meant for **operator ablations** (Reviewer 1 request): we run the same
problem/budget with different index-set operator settings:
  - add_only
  - add_drop
  - add_drop_restart
  - all_clip  (add+drop+restart+clip)

Usage (from repo root):
  python -m experiments.quadratic.run_ablations [options]

Output: CSV in `experiments/output/quadratic_gap_ablations_summary.csv` with one summary row per case: (case, dim, kappa, budget, n_runs,
  mean_relative_optimality, std_relative_optimality, mean_final_gap, std_final_gap,
  mean_initial_gap, mean_grad_evals, std_grad_evals).
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np

# Ensure package is importable when run as script
if __name__ == "__main__" and __file__ is not None:
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from mice.core_impl import MICE
from mice.policy import DropRestartClipPolicy

from .problem import make_problem


class DisabledPolicy(DropRestartClipPolicy):
    """
    A DropRestartClipPolicy that is falsy.

    MICE checks `if self.policy and ...` before calling drop/restart logic, so making
    the policy falsy cleanly disables drop/restart without touching mice code.
    Clipping is still checked, but clip_type defaults to None so it returns immediately.
    """

    def __bool__(self) -> bool:  # pragma: no cover
        return False


def _no_restart(self: MICE, opt_ml: np.ndarray, err_tol: float):
    return False, opt_ml


def run_single(
    *,
    case: str,
    dim: int,
    kappa: float,
    seed: int,
    max_cost: float,
    eps: float = 0.577,
    stop_crit_norm: float = 1e-8,
    max_iter: int = 100_000,
) -> tuple[float, float, float, int, dict[str, int]]:
    """
    Run one SGD-MICE trajectory on the quadratic problem.
    Returns (initial_gap, final_gap, relative_optimality, grad_evals_used, op_counts).
    """
    dobjf, _objf, Eobjf, Edobjf, sampler, optimum, f_opt, L, stepsize = make_problem(
        kappa=kappa, seed=seed, dim=dim
    )

    # --- Operator ablation settings ---
    # We follow Reviewer 1's requested ladder:
    #   add_only -> add_drop -> add_drop_restart -> all_clip
    if case == "add_only":
        # Paper defaults (main.tex around Eq. (2424)):
        #   δ_drop=0.5, δ_rest=0, M_min=5 (general), 50 (restart), max |L_k| = 100.
        # In MICE: M_min = min_batch; restart batch is restart_factor * min_batch.
        policy = DisabledPolicy(
            drop_param=0.5,
            restart_param=0.0,
            max_hierarchy_size=100,
            aggr_cost=0.1,
            clip_type=None,
        )
    elif case == "add_drop":
        # Enable Drop, disable Restart (by patching restart check), disable Clip.
        policy = DropRestartClipPolicy(
            drop_param=0.5,
            restart_param=0.0,
            max_hierarchy_size=100,
            aggr_cost=0.1,
            clip_type=None,
        )
    elif case == "add_drop_restart":
        # Enable Drop and Restart, disable Clip.
        policy = DropRestartClipPolicy(
            drop_param=0.5,
            restart_param=0.0,
            max_hierarchy_size=100,
            aggr_cost=0.1,
            clip_type=None,
        )
    elif case == "all_clip":
        # Enable Drop, Restart, and Clip.
        # Paper: for continuous problems, use clipping \"A\" (argmin over all clip points),
        # which corresponds to MICE clip_type=\"all\" heuristic.
        policy = DropRestartClipPolicy(
            drop_param=0.5,
            restart_param=0.0,
            max_hierarchy_size=100,
            aggr_cost=0.1,
            clip_type="all",
        )
    else:
        raise ValueError(f"Unknown case: {case!r}")

    dF = MICE(
        grad=dobjf,
        sampler=sampler,
        eps=float(eps),
        # Paper defaults: M_min=5 for general iterations, 50 for restarts.
        min_batch=5,
        restart_factor=10,
        stop_crit_norm=float(stop_crit_norm),
        stop_crit_prob=0.05,
        convex=False,
        use_resampling=True,
        max_cost=float(max_cost),
        policy=policy,
    )

    # Patch restart off for the add_drop case, without modifying mice.
    if case == "add_drop":
        dF._check_restart = types.MethodType(_no_restart, dF)  # type: ignore[method-assign]

    x0 = np.full(dim, 20.0, dtype=float)
    X = [x0]
    while (not dF.terminate) and (len(X) < max_iter):
        g = dF.evaluate(X[-1])
        if dF.terminate:
            break
        X.append(X[-1] - stepsize * g)

    # Count operator events from the internal event log.
    # NOTE: MICE logs exactly one event per evaluate() call; if multiple operators happen
    # in one call, the event is overwritten (drop -> restart -> clip), so these counts
    # are conservative for combinations.
    op_counts = {"dropped": 0, "restart": 0, "clip": 0}
    for row in dF.get_log():
        ev = row.get("event")
        if ev in op_counts:
            op_counts[ev] += 1

    gap_0 = float(Eobjf(x0) - f_opt)
    gap_T = float(Eobjf(X[-1]) - f_opt)
    if gap_0 > 0:
        rel_optimality = float(gap_T / gap_0)
    else:
        rel_optimality = 0.0
    grad_evals = int(dF.counter)
    return gap_0, gap_T, rel_optimality, grad_evals, op_counts


def main() -> None:
    p = argparse.ArgumentParser(
        description="Quadratic ablation: fixed budget, n runs, average relative optimality gap decrease"
    )
    p.add_argument("--dim", type=int, default=100, help="Problem dimension")
    p.add_argument("--kappa", type=float, default=100.0, help="Condition number")
    p.add_argument("--max-cost", type=float, default=100_000, help="Max gradient evaluations per run")
    p.add_argument("--runs", type=int, default=50, help="Number of runs to average")
    p.add_argument("--seed", type=int, default=1, help="Base seed (per-run seed = seed + run_index)")
    p.add_argument(
        "--cases",
        type=str,
        default="add_only,add_drop,add_drop_restart,all_clip",
        help="Comma- or space-separated list of cases to run "
        "(default: add_only,add_drop,add_drop_restart,all_clip)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV path (default: <outdir>/quadratic_gap_ablations_summary.csv)",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "output"),
        help="If set, write CSV to this directory (overrides --csv dir part)",
    )
    args = p.parse_args()

    n_runs = max(1, args.runs)

    case_tokens = args.cases.replace(",", " ").split()
    cases = [c.strip() for c in case_tokens if c.strip()]
    if not cases:
        raise ValueError("No cases selected. Use --cases to specify cases to run.")

    # Output path
    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / "quadratic_gap_ablations_summary.csv"
    elif args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        csv_path = Path("quadratic_gap_ablations_summary.csv")

    # Write one summary row per case (paper table).
    import csv as csv_module
    rows = []
    for case in cases:
        runs_data = []
        print(f"\nCase: {case} (dim={args.dim}, kappa={args.kappa}, budget={args.max_cost}, runs={n_runs})")
        for r in range(n_runs):
            seed = args.seed + r
            gap_0, gap_T, rel_optimality, grad_evals, op_counts = run_single(
                case=case,
                dim=args.dim,
                kappa=args.kappa,
                seed=seed,
                max_cost=args.max_cost,
                stop_crit_norm=1e-8,
                max_iter=100_000,
            )
            runs_data.append(
                {
                    "initial_gap": gap_0,
                    "final_gap": gap_T,
                    "relative_optimality": rel_optimality,
                    "grad_evals": grad_evals,
                    "num_drop_events": int(op_counts.get("dropped", 0)),
                    "num_restart_events": int(op_counts.get("restart", 0)),
                    "num_clip_events": int(op_counts.get("clip", 0)),
                }
            )
            print(
                f"  run {r + 1}/{n_runs}: rel_optimality={rel_optimality:.4f}, "
                f"final_gap={gap_T:.2e}, grads={grad_evals}, "
                f"drop={op_counts.get('dropped', 0)}, restart={op_counts.get('restart', 0)}, clip={op_counts.get('clip', 0)}"
            )

        rel_optimalitys = np.array([d["relative_optimality"] for d in runs_data], dtype=float)
        final_gaps = np.array([d["final_gap"] for d in runs_data], dtype=float)
        initial_gaps = np.array([d["initial_gap"] for d in runs_data], dtype=float)
        grad_evals_arr = np.array([d["grad_evals"] for d in runs_data], dtype=float)
        drop_events_arr = np.array([d["num_drop_events"] for d in runs_data], dtype=float)
        restart_events_arr = np.array([d["num_restart_events"] for d in runs_data], dtype=float)
        clip_events_arr = np.array([d["num_clip_events"] for d in runs_data], dtype=float)

        mean_rel_optimality = float(np.mean(rel_optimalitys))
        std_rel_optimality = float(np.std(rel_optimalitys, ddof=1)) if n_runs > 1 else 0.0
        mean_final_gap = float(np.mean(final_gaps))
        std_final_gap = float(np.std(final_gaps, ddof=1)) if n_runs > 1 else 0.0
        mean_initial_gap = float(np.mean(initial_gaps))
        mean_grad_evals = float(np.mean(grad_evals_arr))
        std_grad_evals = float(np.std(grad_evals_arr, ddof=1)) if n_runs > 1 else 0.0
        mean_drop_events = float(np.mean(drop_events_arr))
        mean_restart_events = float(np.mean(restart_events_arr))
        mean_clip_events = float(np.mean(clip_events_arr))

        row = {
            "case": case,
            "dim": args.dim,
            "kappa": args.kappa,
            "budget": args.max_cost,
            "n_runs": n_runs,
            "mean_relative_optimality": mean_rel_optimality,
            "std_relative_optimality": std_rel_optimality,
            "mean_final_gap": mean_final_gap,
            "std_final_gap": std_final_gap,
            "mean_initial_gap": mean_initial_gap,
            "mean_grad_evals": mean_grad_evals,
            "std_grad_evals": std_grad_evals,
            "mean_drop_events": mean_drop_events,
            "mean_restart_events": mean_restart_events,
            "mean_clip_events": mean_clip_events,
        }
        rows.append(row)

        print(f"  Summary ({case}): mean_relative_optimality = {mean_rel_optimality:.4f} ± {std_rel_optimality:.4f}")
        print(f"                  mean_final_gap = {mean_final_gap:.2e} ± {std_final_gap:.2e}")
        print(f"                  mean events: drop={mean_drop_events:.2f}, restart={mean_restart_events:.2f}, clip={mean_clip_events:.2f}")

    with open(csv_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
