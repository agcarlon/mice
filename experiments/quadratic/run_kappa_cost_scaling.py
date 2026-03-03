from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mice.core_impl import MICE
from mice.policy import DropRestartClipPolicy

from .problem import make_problem


@dataclass(frozen=True, slots=True)
class MethodResult:
    kappa: float
    cost: float
    terminate_reason: str | None


class _DisabledPolicy(DropRestartClipPolicy):
    def __bool__(self) -> bool:  # pragma: no cover
        return False


def _run_until_tol(
    *,
    kappa: float,
    seed: int,
    eps: float,
    tol: float,
    max_cost: float,
    max_iter: int,
    mode: str,
    use_resampling: bool,
    require_tol: bool,
) -> MethodResult:
    dobjf, _objf, _Eobjf, _Edobjf, sampler, _optimum, _f_opt, _L, stepsize = make_problem(
        kappa=kappa, seed=seed, dim=2
    )

    if mode == "vanilla_mice":
        policy = _DisabledPolicy(clip_type=None)
    elif mode == "mice_ops":
        policy = DropRestartClipPolicy(
            drop_param=0.5,
            restart_param=0.0,
            max_hierarchy_size=100,
            aggr_cost=0.1,
            clip_type="all",
        )
    elif mode == "sgd_a":
        policy = DropRestartClipPolicy(
            drop_param=0.0,
            restart_param=0.0,
            max_hierarchy_size=1,
            aggr_cost=0.0,
            clip_type=None,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    dF = MICE(
        grad=dobjf,
        sampler=sampler,
        eps=float(eps),
        min_batch=10,
        restart_factor=10,
        stop_crit_norm=float(tol),
        stop_crit_prob=0.05,
        convex=True,
        use_resampling=bool(use_resampling),
        max_cost=float(max_cost),
        policy=policy,
    )
    # This experiment only needs final cost/termination reason; keeping the full
    # per-iteration recorder can blow up memory for tight tolerances.
    dF.recorder = None

    x = np.array([20.0, 50.0], dtype=float)
    for _k in range(int(max_iter)):
        if mode == "sgd_a":
            dF.force_restart = True
        g = dF.evaluate(x)
        if dF.terminate:
            break
        x = x - stepsize * g
        if (_k + 1) % 50_000 == 0:
            print(f"  iter={_k+1} cost={dF.counter:.0f} reason={dF.terminate_reason!r}")

    if not dF.terminate:
        dF.terminate = True
        dF.terminate_reason = "max_iter"

    if require_tol and dF.terminate_reason != "stop_crit":
        raise RuntimeError(
            f"Did not reach stop criterion for kappa={kappa:g} mode={mode!r}. "
            f"terminate_reason={dF.terminate_reason!r}. "
            f"Increase --max-iter (and ensure --max-cost is inf)."
        )

    return MethodResult(
        kappa=float(kappa),
        cost=float(dF.counter),
        terminate_reason=getattr(dF, "terminate_reason", None),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Cost vs kappa scaling: vanilla SGD-MICE vs operator SGD-MICE vs SGD-A.")
    p.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parents[1] / "output"))
    p.add_argument("--quick", action="store_true", help="Reduce runtime (not paper settings).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eps", type=float, default=float(np.sqrt(1.0 / 3.0)))
    p.add_argument("--tol", type=float, default=1e-1, help="Stopping criterion on ||∇F||^2.")
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write output even if some runs do not reach tol (otherwise raises).",
    )
    p.add_argument(
        "--resampling",
        action="store_true",
        help="Use gradient-resampling norm estimator (paper uses resampling for the stopping-consistency figure; "
        "this kappa-scaling test defaults to plain norm mode).",
    )
    p.add_argument(
        "--kappas",
        type=str,
        default="2, 4, 8, 16, 32, 64, 128",
        help="Comma- or space-separated kappa values.",
    )
    p.add_argument("--max-cost", type=float, default=float("inf"), help="Optional cap on gradient evaluations.")
    p.add_argument("--max-iter", type=int, default=int(1e7), help="Hard cap on iterations.")
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    def _parse_floats(s: str) -> list[float]:
        toks = [t for t in s.replace(",", " ").split() if t.strip()]
        return [float(t) for t in toks]

    kappas = _parse_floats(args.kappas)
    if args.quick:
        kappas = kappas[:4]
        tol = 1e-4
        max_cost = float(args.max_cost)
        max_iter = 50_000
    else:
        tol = float(args.tol)
        max_cost = float(args.max_cost)
        max_iter = int(args.max_iter)

    require_tol = not bool(args.allow_incomplete)
    rows = []
    for kappa in kappas:
        for mode, label in (
            ("vanilla_mice", "vanilla_SGD-MICE"),
            ("mice_ops", "SGD-MICE"),
            ("sgd_a", "SGD-A"),
        ):
            r = _run_until_tol(
                kappa=float(kappa),
                seed=int(args.seed),
                eps=float(args.eps),
                tol=tol,
                max_cost=max_cost,
                max_iter=max_iter,
                mode=mode,
                use_resampling=bool(args.resampling),
                require_tol=require_tol,
            )
            rows.append(
                {
                    "kappa": r.kappa,
                    "method": label,
                    "cost": r.cost,
                    "terminate_reason": r.terminate_reason,
                    "tol": tol,
                    "eps": float(args.eps),
                }
            )
            print(f"kappa={kappa:g} {label:14s} cost={r.cost:.0f} terminate_reason={r.terminate_reason!r}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "kappa_cost_scaling.csv", index=False)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for label, color in (
        ("vanilla_SGD-MICE", "C2"),
        ("SGD-MICE", "C0"),
        ("SGD-A", "C1"),
    ):
        d = df[df["method"] == label].sort_values("kappa")
        ax.loglog(d["kappa"], d["cost"], marker="o", label=label, color=color)

    # Reference lines (scaled to roughly align with the middle point)
    ks = np.asarray(sorted(set(kappas)), dtype=float)
    if len(ks):
        k0 = ks[len(ks) // 2]
        # pick a baseline y at k0 from SGD-A if present, else from SGD-MICE.
        y0 = float(df[(df["kappa"] == k0) & (df["method"] == "SGD-A")]["cost"].iloc[0]) if any(
            (df["kappa"] == k0) & (df["method"] == "SGD-A")
        ) else float(df[(df["kappa"] == k0) & (df["method"] == "SGD-MICE")]["cost"].iloc[0])
        ax.loglog(ks, y0 * (ks / k0) ** 2, "k--", lw=1.0, label=r"$\mathcal{O}(\kappa^2)$")
        ax.loglog(ks, y0 * (ks / k0), "k-.", lw=1.0, label=r"$\mathcal{O}(\kappa)$")

    ax.set_xlabel(r"Condition number $\kappa$")
    ax.set_ylabel(r"Gradient sampling cost $\mathcal{C}$")
    ax.set_title("Cost scaling vs condition number")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    outpath = outdir / "kappa_test_w_sgd_a.pdf"
    fig.savefig(str(outpath))
    plt.close(fig)
    print(f"Wrote: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
