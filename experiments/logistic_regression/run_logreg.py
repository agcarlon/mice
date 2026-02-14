from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, softplus

from mice import plot_mice
from mice.core_impl import MICE
from mice.policy import DropRestartClipPolicy

from .datasets import ensure_dataset_npy
from .load_data import compute_L_hessian, load_dataset_shuffled
from .configs import get_config


def sgd_mice(
    dataset: str,
    eps: float,
    Lambda: float,
    epochs_total: int,
    *,
    seed: int = 0,
    use_resampling: bool = False,
    for_optimum: bool = False,
):
    """
    Port of `mice_numerics/logistic_regression/logreg/sgd_mice.py`, but using MICE.

    Expects data files:
    - `<base>/<dataset>/<dataset>.npy`
    - `<base>/<dataset>/Optimum.npy` (unless for_optimum=True)

    When for_optimum=True, Optimum.npy is not required; the run returns (W_final, loss_final)
    for use as the reference optimum (e.g. with a larger budget and dedicated seed).
    """
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / dataset
    data_path = ensure_dataset_npy(dataset, base_dir=base_dir)
    opt_path = dataset_dir / "Optimum.npy"
    if not for_optimum and not opt_path.exists():
        raise FileNotFoundError(
            f"Missing optimum file: {opt_path}\n"
            f"Generate it with: python -m experiments.logistic_regression.compute_optimum --dataset {dataset}"
        )

    if not for_optimum:
        print(f"Solving logistic regression of {dataset} dataset using SGD-MICE")

    run_dir = dataset_dir / "sgd_mice" / str(seed) if not for_optimum else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    name = (run_dir / "sgd_mice") if run_dir is not None else None

    t_load = time()
    X, Y, _, thetas_mat, datasize, n_features = load_dataset_shuffled(data_path, seed)
    X = np.asarray(X, dtype=float, copy=False)
    Y = np.asarray(Y, dtype=float, copy=False)
    print(f"Load time: {time() - t_load:.3f}s")

    W0 = np.zeros(shape=(n_features,), dtype=float)
    opt_W = np.load(str(opt_path)).astype(float, copy=False) if not for_optimum else None

    t_L = time()
    L = compute_L_hessian(X, datasize, Lambda)
    print(f"L: {L:.6g} (computed in {time() - t_L:.3f}s)")

    def sigmoid(z):
        # stable sigmoid
        return expit(z)

    def logloss_full(W: np.ndarray) -> float:
        # Paper objective: log(1 + exp(-y <w,x>)) + (lambda/2)||w||^2; softplus stable in float64
        z = -Y * (X @ W)
        return float(np.mean(softplus(z)) + 0.5 * Lambda * (W @ W))

    def lossgrad(W: np.ndarray, batch: np.ndarray) -> np.ndarray:
        # batch: (m, 1+d), y in col 0, x in 1:
        y = batch[:, 0]
        xb = batch[:, 1:]
        # grad = -y * x * sigmoid(-y <w,x>) + Lambda*w
        gr = (sigmoid(-y * (xb @ W)) * (-y))[:, None] * xb
        gr += (Lambda * W)[None, :]
        return gr

    start_loss = logloss_full(W0)
    if not for_optimum:
        opt_loss = logloss_full(opt_W)
        print(f"start loss: {start_loss:.6f}, opt loss: {opt_loss:.6f}")

    # Get dataset-specific configuration
    config = get_config(dataset)
    
    policy = DropRestartClipPolicy(
        drop_param=config["drop_param"],
        restart_param=config["restart_param"],
        max_hierarchy_size=config["max_hierarchy_size"],  # 100 for mushrooms/gisette, 1000 for higgs
        aggr_cost=0.1,
        clip_type=None,
        clip_every=0,
    )

    df = MICE(
        grad=lossgrad,
        sampler=thetas_mat,
        eps=float(eps),
        restart_factor=config["restart_factor"],  # min_batch * restart_factor = 5 * 10 = 50 for restarts
        min_batch=config["min_batch"],
        max_cost=float(epochs_total * datasize),
        convex=True,
        policy=policy,
        use_resampling=bool(use_resampling),
        stop_crit_norm=0.0,
        re_part=config["re_part"],
        re_quantile=config["re_quantile"],
        re_tot_cost=config["re_tot_cost"],
        re_min_n=config["re_min_n"],
    )
    # Match v1's deterministic behavior: MICE internals use randomness for the finite sampler start.
    df.rng = np.random.default_rng(seed)

    stepsize = 1.0 / L
    k = 1
    W = [W0]

    epoch = 0
    times_epoch = [1e-2]
    losses = [start_loss]
    epochs = [0]
    kepoch = [1]
    counter = [1]

    t_epoch = time()
    np.random.seed(0)
    while not df.terminate:
        k += 1
        grad = df.evaluate(W[-1])
        if df.terminate:
            break
        W.append(W[-1] - stepsize * grad)
        if df.counter >= (epoch + 1) * datasize:
            times_epoch.append(time() - t_epoch)
            loss_k = logloss_full(W[-1])
            losses.append(loss_k)
            counter.append(df.counter)
            # df.counter can overshoot the nominal max_cost by up to one batch;
            # clamp so the fixed-size epoch arrays (0..epochs_total) stay in-bounds.
            epoch = int(min(np.floor(df.counter / datasize), epochs_total))
            epochs.append(epoch)
            kepoch.append(k)
            t_epoch = time()

    times_epoch.append(time() - t_epoch)
    runtime = np.array(times_epoch)
    loss_k = logloss_full(W[-1])
    losses.append(loss_k)
    counter.append(df.counter)
    epochs.append(int(epochs_total))
    kepoch.append(k)

    if for_optimum:
        # For sanity/debugging on large datasets: show how much budget was actually used.
        max_cost = float(epochs_total * datasize)
        print(
            "SGD-MICE optimum run summary: "
            f"datasize={datasize} epochs={epochs_total} max_cost={max_cost:.0f} "
            f"num_grads={int(df.counter)} terminate_reason={getattr(df, 'terminate_reason', None)!r}"
        )
        return (np.asarray(W[-1], dtype=float), float(logloss_full(W[-1])))

    # Chain size plot from per-iteration MICE log
    log_iter = pd.DataFrame(df.get_log())
    if len(log_iter):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax = plot_mice(log_iter, ax, x="iteration", y="hier_length", style="plot", legend=False)
        chain_diff = log_iter["hier_length"].diff()
        clip_iter = log_iter[(chain_diff < 0) & (log_iter["event"] != "restart") & (log_iter["event"] != "end")].index
        for it in clip_iter:
            ax.plot(log_iter[it - 1 : it + 1]["iteration"], log_iter[it - 1 : it + 1]["hier_length"], "r")
        ax.plot([], [], "r", label="Clipping")
        ax.legend(bbox_to_anchor=(1.02, 1, 1.02, 0), ncol=1, loc="upper left")
        ax.set_title(f"Logistic regression - {dataset} dataset")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$|\mathcal{L}_k|$")

        # Match v1 plot: add a small stats box (adds/drops/restarts/clips)
        try:
            texts = [
                f"Add: {int((log_iter['event'] == 'add').sum())}",
                f"Drops: {int((log_iter['event'] == 'dropped').sum())}",
                f"Restarts: {int((log_iter['event'] == 'restart').sum())}",
                f"Clips: {int(len(clip_iter))}",
            ]
            props = dict(boxstyle="round", facecolor="white", alpha=0.2)
            ax.text(
                1.135,
                0.35,
                "\n".join(texts),
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="center",
                bbox=props,
            )
        except Exception:
            # Don't fail the run if the log format changes
            pass

        fig.tight_layout()
        fig.savefig(str(name) + "_chain_size.pdf")
        plt.close(fig)

    # Epoch-level summary (matches v1 format expected by plots_losses.py)
    data = np.full((5, epochs_total + 1), fill_value=np.nan)
    data[0] = np.arange(epochs_total + 1)
    data[1, epochs] = losses
    data[2, epochs] = counter
    data[3, epochs] = kepoch
    data[4, epochs] = runtime

    log = pd.DataFrame(data.T, columns=["epochs", "losses", "num_grads", "iteration", "runtime"])
    log["rel_loss"] = (log["losses"] - opt_loss) / (start_loss - opt_loss)
    log.to_pickle(str(name) + ".pkl")

    # Per-run "usual" relative optimality-gap plots (mirrors plots_losses.py, but saved in this seed folder)
    try:
        plot_df = log.copy()
        # pkls store per-epoch time deltas; convert to cumulative for plotting
        plot_df["runtime"] = plot_df["runtime"].fillna(0).cumsum()
        plot_df["event"] = None

        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
        axs[0] = plot_mice(plot_df, axs[0], x="num_grads", y="rel_loss", markers=False, color="C0", style="semilogy")
        opt_gap_string = r"$\frac{F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)}{F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi_0})}$"
        axs[0].set_ylabel(opt_gap_string)
        axs[0].set_xlabel(r"$\#$ grads")
        axs[0].set_title(f"Logistic regression - {dataset} dataset")

        axs[1] = plot_mice(plot_df, axs[1], x="iteration", y="rel_loss", markers=False, color="C0", style="semilogy")
        axs[1].set_ylabel(opt_gap_string)
        axs[1].set_xlabel("Iteration")

        axs[2] = plot_mice(plot_df, axs[2], x="runtime", y="rel_loss", markers=False, color="C0", style="semilogy")
        axs[2].set_ylabel(opt_gap_string)
        axs[2].set_xlabel("Runtime (s)")

        for ax in axs:
            for ln in ax.lines:
                ln.set_marker("o")
                ln.set_ms(3)

        fig.tight_layout()
        fig.savefig(str(name) + "_rel_loss.pdf")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: failed to save per-run rel_loss plots: {e}")

    print(
        "Done. "
        f"rel_loss(last)={float(log['rel_loss'].dropna().iloc[-1]):.3e}, "
        f"runtime(total)={float(np.sum(runtime)):.3f}s"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--eps", type=float, default=float(1 / np.sqrt(3)))
    p.add_argument("--lambda", dest="Lambda", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resampling", action="store_true")
    args = p.parse_args()

    config = get_config(args.dataset)
    lambda_reg = float(args.Lambda) if args.Lambda is not None else float(config["lambda_reg"])
    epochs = int(args.epochs) if args.epochs is not None else int(config["epochs"])

    sgd_mice(args.dataset, args.eps, lambda_reg, epochs, seed=args.seed, use_resampling=args.resampling)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
