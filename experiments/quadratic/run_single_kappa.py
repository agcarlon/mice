from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mice import plot_mice
from mice.core_impl import MICE

from .problem import make_problem


def run(
    *,
    eps_rel: float,
    kappa: float,
    seed: int,
    use_resampling: bool,
    outdir: Path,
    max_cost: float = float("inf"),
    stop_crit_norm: float = 1e-8,
    max_iter: int = int(1e7),
):
    dobjf, _objf, Eobjf, Edobjf, sampler, optimum, f_opt, L, stepsize = make_problem(kappa=kappa, seed=seed)

    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / "SGD_MICE"

    dF = MICE(
        grad=dobjf,
        sampler=sampler,
        eps=float(eps_rel),
        min_batch=10,
        restart_factor=10,
        stop_crit_norm=float(stop_crit_norm),
        stop_crit_prob=0.05,
        convex=False,
        use_resampling=bool(use_resampling),
        # v1/paper script typically runs without a max_cost cap; keep parity by default.
        max_cost=float(max_cost),
    )

    X = [np.array([20.0, 50.0])]
    grads = []
    bias = []
    stat_err = []

    while (not dF.terminate) and (len(X) < int(max_iter)):
        g = dF.evaluate(X[-1])
        if dF.terminate:
            break
        grads.append(g)
        X.append(X[-1] - stepsize * g)
        # MICE currently exposes these as internal methods; for numerics parity we use them here
        bias.append(dF._compute_bias())
        stat_err.append(dF._compute_statistical_error())

    # If we stopped due to max_iter (not via MICE termination), mark it explicitly and
    # append a final log row so downstream analysis can rely on an 'end' event.
    if not dF.terminate and len(X) >= int(max_iter):
        dF.terminate = True
        dF.terminate_reason = "max_iter"
        g_hat = dF._g_hat if dF._g_hat is not None else np.zeros_like(X[-1])
        dF._record(event="end", g_hat=g_hat)  # type: ignore[attr-defined]

    Fs = [Eobjf(x) - f_opt for x in X]
    true_grads = np.vstack([Edobjf(x) for x in X])
    grads_arr = np.vstack(grads) if len(grads) else np.zeros((0, true_grads.shape[1]))

    log = pd.DataFrame(dF.get_log())
    # v1-compatible names used by plot_mice and existing paper scripts
    if "last_v" in log.columns:
        log["vl"] = log["last_v"]
    if "grad_norm" in log.columns:
        log["grad_norm"] = log["grad_norm"] ** 2
    log["opt_gap"] = Fs[: len(log)]
    log["dist_to_opt"] = np.linalg.norm(np.vstack(X[: len(log)]) - optimum, axis=1) ** 2

    # Relative error curve:
    # MICE logs one row per evaluate() call, including a terminal 'end' row. Our loop
    # does not necessarily append a gradient for that final row, so we align by
    # filling what we have and leaving the remainder as NaN.
    log["rel_error"] = np.nan
    n_rel = int(min(len(grads_arr), len(log)))
    if n_rel > 0:
        true_grads_used = true_grads[:n_rel]
        rel = np.linalg.norm(grads_arr[:n_rel] - true_grads_used, axis=1) ** 2 / np.linalg.norm(true_grads_used, axis=1) ** 2
        log.loc[log.index[:n_rel], "rel_error"] = rel

    # Bias/stat error (v1 paper plots use these)
    if len(bias) < len(log):
        # pad to length
        bias = bias + [bias[-1] if bias else 0.0] * (len(log) - len(bias))
        stat_err = stat_err + [stat_err[-1] if stat_err else 0.0] * (len(log) - len(stat_err))
    log["bias"] = bias[: len(log)]
    log["stat_err"] = stat_err[: len(log)]
    log["rel_bias"] = log["bias"] / np.maximum(log["grad_norm"], 1e-300)
    log["rel_stat_err"] = log["stat_err"] / np.maximum(log["grad_norm"], 1e-300)

    log.to_pickle(str(stem) + ".pkl")

    # --- Plots (ported from `mice_numerics/quad_c.py`) ---
    mc_conv_grads = [1e6, 1e8]
    mc_conv_loss2 = [1e-4, 1e-4 * 100 ** -1]
    mc_conv_dist = [1e-4, 1e-4 * 100 ** -1]
    mc_conv_norm = [1e-4, 1e-4 * 100 ** -1]

    # per gradient evaluations
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axs[0] = plot_mice(log, axs[0], x="num_grads", y="opt_gap", legend=False)
    axs[0].set_ylabel(r"$F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)$")
    axs[0].plot(mc_conv_grads, mc_conv_loss2, "k-.", label=r"$\mathcal{O}(\mathcal{C}_k^{-1})$")
    axs[0].legend()

    axs[1] = plot_mice(log, axs[1], x="num_grads", y="dist_to_opt", legend=False)
    axs[1].set_ylabel(r"$\left\|\boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\|^2$")
    axs[1].plot(mc_conv_grads, mc_conv_dist, "k-.")

    axs[2] = plot_mice(log, axs[2], x="num_grads", y="grad_norm", legend=False)
    axs[2].axhline(dF.stop_crit_norm, c="k", ls="--")
    axs[2].plot(mc_conv_grads, mc_conv_norm, "k-.")
    axs[2].set_ylabel(r"$\left\|\nabla_{\boldsymbol{\xi}} \mathcal{F}_k\right\|^2$")

    axs[3] = plot_mice(log, axs[3], x="num_grads", y="iteration", legend=False, style="semilogx")
    axs[3].set_ylabel("Iteration")
    axs[3].set_xlabel(r"$\mathcal{C}_k$")

    plt.tight_layout()
    plt.savefig(str(stem) + "_all_per_grads.pdf")
    plt.close(fig)

    # per iteration
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0] = plot_mice(log, axs[0], x="iteration", y="opt_gap", style="semilogy")
    axs[0].set_ylabel(r"$F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)$")
    axs[0].legend()

    axs[1] = plot_mice(log, axs[1], x="iteration", y="dist_to_opt", legend=False, style="semilogy")
    axs[1].set_ylabel(r"$\left\|\boldsymbol{\xi}_k - \boldsymbol{\xi}^*\right\|^2$")

    axs[2] = plot_mice(log, axs[2], x="iteration", y="grad_norm", legend=False, style="semilogy")
    axs[2].axhline(dF.stop_crit_norm, c="k", ls="--")
    axs[2].set_ylabel(r"$\left\|\nabla_{\boldsymbol{\xi}} \mathcal{F}_k\right\|^2$")
    axs[2].set_xlabel("Iteration")

    plt.tight_layout()
    plt.savefig(str(stem) + "_all_per_iter.pdf")
    plt.close(fig)

    # chain size plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax = plot_mice(log, ax, x="iteration", y="hier_length", style="plot", legend=False)
    chain_diff = log["hier_length"].diff()
    clip_iter = log[(chain_diff < 0) & (log["event"] != "restart")].index
    for it in clip_iter:
        ax.plot(log[it - 1 : it + 1]["iteration"], log[it - 1 : it + 1]["hier_length"], "r")
    ax.plot([], [], "r", label="Clipping")
    ax.legend(bbox_to_anchor=(1.02, 1, 1.02, 0), ncol=1, loc="upper left")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$|\mathcal{L}_k|$")
    fig.tight_layout()
    plt.savefig(str(stem) + "_chain_size.pdf")
    plt.close(fig)

    # --- Manuscript figure: error/vkk/chain size (single run) ---
    # The manuscript uses the same run data as the per-iter/per-grads plots above.
    # We follow the caption order: |L_k|, true squared relative error, empirical rel. error (bias/stat),
    # and V_{k,k} vs iteration.
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    # Trying to squeeze index set size in this plot (ported from old paper code)
    axs[0] = plot_mice(log, axs[0], x="iteration", y="hier_length", style="plot", legend=False)
    chain_diff = log["hier_length"].diff()
    clip_iter = log[(chain_diff < 0) & (log["event"] != "restart")].index
    for it in clip_iter:
        axs[0].plot(log[it - 1 : it + 1]["iteration"], log[it - 1 : it + 1]["hier_length"], "r")
    axs[0].plot([], [], "r", label="Clipping")

    axs[0].legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
        ncol=3,
        loc="lower center",
        mode="expand",
    )
    axs[0].set_ylabel(r"$| \mathcal{L}_{k} |$")
    axs[0].set_xlim([-5, len(log) + 5])

    texts = [
        f"Add: {len(log[log['event'] == 'add'])}\n",
        f"Drop: {len(log[log['event'] == 'dropped'])}\n",
        f"Restart: {len(log[log['event'] == 'restart'])}\n",
        f"Clip: {len(clip_iter)}",
    ]
    props = dict(boxstyle="round", facecolor="white", alpha=1.0, fill=True, edgecolor=".85")
    axs[0].text(
        0.87,
        0.1,
        "".join(texts),
        transform=axs[0].transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=props,
    )

    axs[1] = plot_mice(log, axs[1], x="iteration", y="rel_error", style="semilogy", legend=False)
    axs[1].set_ylabel(
        r"$\|\nabla_{\boldsymbol{\xi}} \mathcal{F}_k - \nabla_{\boldsymbol{\xi}} F_k\|^2 / "
        r"\|\nabla_{\boldsymbol{\xi}} F_k\|^2$"
    )
    axs[1].plot(log["iteration"], np.full(len(log), eps_rel**2), "k--")
    x_text = float(log["iteration"].iloc[min(30, max(len(log) - 1, 0))]) if len(log) else 30.0
    axs[1].text(x_text, (eps_rel**2) * 1.15, r"$\epsilon^2$", verticalalignment="baseline")
    axs[1].set_ylim(top=np.maximum(axs[1].get_ylim()[1], (eps_rel**2) * 5))

    # Empirical relative error shows shaded regions (bias/statistical error)
    axs[2].fill_between(log["iteration"], 0, log["rel_bias"], color="r", alpha=0.5, label="Bias")
    axs[2].fill_between(
        log["iteration"],
        log["rel_bias"],
        log["rel_bias"] + log["rel_stat_err"],
        color="b",
        alpha=0.5,
        label="Statistical Error",
    )
    axs[2].set_ylabel(r"$(\sum_{\ell} V_{\ell, k} / M_{\ell, k}) / \|\mathcal{F}_k\|^{2}$")
    axs[2].plot(log["iteration"], np.full(len(log), eps_rel**2), "k--", label=r"$\epsilon^2$")
    axs[2].set_ylim(bottom=0)
    axs[2].legend(loc="lower right")

    axs[3] = plot_mice(log, axs[3], x="iteration", y="vl", style="semilogy", legend=False)
    axs[3].set_ylabel(r"$V_{k,k}$")
    axs[3].set_xlabel(r"Iteration")

    fig.tight_layout()
    plt.savefig(str(outdir / "SGD_MICE_err_vkk_chain_size.pdf"))
    plt.close(fig)

    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps-rel", type=float, default=float(np.sqrt(1 / 3)))
    p.add_argument("--kappa", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resampling", action="store_true", help="Enable resampling-based tol/stopping")
    p.add_argument("--max-cost", type=float, default=float("inf"), help="Optional cap on gradient evaluations (default: inf)")
    p.add_argument("--stop-crit-norm", type=float, default=1e-8, help="Stopping tolerance on ||∇F||^2 (default: 1e-8)")
    p.add_argument("--max-iter", type=int, default=int(1e7), help="Hard cap on iterations (default: 1e7)")
    p.add_argument(
        "--ensure-tol",
        action="store_true",
        help="Fail if the run does not terminate via stop criterion (i.e., reaches tol).",
    )
    p.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parents[1] / "output" / "QuadC_"))
    args = p.parse_args()

    outdir = Path(args.outdir)
    # keep old script behavior: create folder and subfolder
    # We forward max_cost / stop_crit_norm / max_iter by temporarily overriding inside run()
    # to keep the core logic close to the original quad_c.py.
    log = run(
        eps_rel=args.eps_rel,
        kappa=args.kappa,
        seed=args.seed,
        use_resampling=args.resampling,
        outdir=outdir,
        max_cost=args.max_cost,
        stop_crit_norm=args.stop_crit_norm,
        max_iter=args.max_iter,
    )
    if args.ensure_tol:
        last_reason = None
        if len(log):
            last_reason = log.iloc[-1].get("terminate_reason")
        if last_reason != "stop_crit":
            raise RuntimeError(
                f"Did not terminate by stop criterion. terminate_reason={last_reason!r}. "
                f"Increase --max-cost/--max-iter (or set them to inf) to run until tol."
            )
    print(f"Done. Outputs written under: {outdir}")


if __name__ == "__main__":
    # Make plots deterministic enough for comparisons
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
