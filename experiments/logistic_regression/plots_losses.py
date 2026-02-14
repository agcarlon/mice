from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mice import plot_mice


def plots_losses(dataset: str, *, base_dir: Path, out_dir: Path | None = None):
    # Use seed=0 runs by convention (mirrors v1 usage).
    names = [
        base_dir / dataset / "sgd_mice" / "0" / "sgd_mice.pkl",
        base_dir / dataset / "sag" / "0" / "sag.pkl",
        base_dir / dataset / "saga" / "0" / "saga.pkl",
        base_dir / dataset / "sarah" / "0" / "sarah.pkl",
        base_dir / dataset / "svrg" / "0" / "svrg.pkl",
    ]
    labels = ("SGD-MICE", "SAG", "SAGA", "SARAH", "SVRG")
    colors = ["C0", "C1", "C2", "C3", "C4"]

    Data = [pd.read_pickle(str(name)) for name in names]
    for d in Data:
        # pkls store per-epoch time *deltas*; convert to cumulative runtime for plotting
        if "runtime" in d.columns:
            d["runtime"] = d["runtime"].fillna(0).cumsum()
        d["event"] = None

    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    for Data_, color in zip(Data, colors):
        axs[0] = plot_mice(Data_, axs[0], x="num_grads", y="rel_loss", markers=False, color=color, style="semilogy")

    opt_gap_string = r"$\frac{F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)}{F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi_0})}$"
    axs[0].set_ylabel(opt_gap_string)
    axs[0].set_xlabel(r"$\#$ grads")
    axs[0].set_title(f"Logistic regression - {dataset} dataset")
    axs[0].legend(labels)

    for Data_, color in zip(Data, colors):
        axs[1] = plot_mice(Data_, axs[1], x="iteration", y="rel_loss", markers=False, color=color, style="semilogy")
    axs[1].set_ylabel(opt_gap_string)
    axs[1].set_xlabel("Iteration")

    for Data_, color in zip(Data, colors):
        axs[2] = plot_mice(Data_, axs[2], x="runtime", y="rel_loss", markers=False, color=color, style="semilogy")
    axs[2].set_ylabel(opt_gap_string)
    axs[2].set_xlabel("Runtime (s)")

    for ax in axs:
        for ln in ax.lines:
            ln.set_marker("o")
            ln.set_ms(3)

    fig.tight_layout()
    target_dir = out_dir if out_dir is not None else base_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(target_dir / f"relative_loss_gap_{dataset}_.pdf"))


def plots_losses_statistics(dataset: str, seeds, *, base_dir: Path, out_dir: Path | None = None):
    interval = 0.95
    low = (1 - interval) / 2
    high = 1 - low

    names = ["sgd_mice", "sag", "saga", "sarah", "svrg"]
    labels = ("SGD-MICE", "SAG", "SAGA", "SARAH", "SVRG")
    colors = ["C0", "C1", "C2", "C3", "C4"]

    Data = []
    for name in names:
        losses = []
        iters = []
        times = []
        costs = []
        for seed in seeds:
            p = base_dir / dataset / name / str(seed) / f"{name}.pkl"
            data_seed = pd.read_pickle(str(p))
            iters.append(data_seed["iteration"])
            times.append(data_seed["runtime"])
            costs.append(data_seed["num_grads"])
            losses.append(data_seed["rel_loss"])
        losses_pd = pd.concat(losses, axis=1, ignore_index=True)
        losses_pd = (losses_pd.T.reset_index()).T

        iters_pd = pd.concat(iters, axis=1, ignore_index=True)
        iters_pd = (iters_pd.T.reset_index()).T

        times_pd = pd.concat(times, axis=1, ignore_index=True)
        times_pd = (times_pd.T.reset_index()).T

        costs_pd = pd.concat(costs, axis=1, ignore_index=True)
        costs_pd = (costs_pd.T.reset_index()).T

        low_quantile = losses_pd.T.quantile(low)[1:]
        high_quantile = losses_pd.T.quantile(high)[1:]
        mean = losses_pd.T.mean()[1:]

        Data.append(
            pd.DataFrame(
                {
                    "low": low_quantile,
                    "mean": mean,
                    "high": high_quantile,
                    "iteration": iters_pd.T.mean()[1:],
                    "runtime": np.cumsum(times_pd.T.mean()[1:]),
                    "num_grads": costs_pd.T.mean()[1:],
                }
            )
        )

    for d in Data:
        d["event"] = None

    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    for Data_, color, label in zip(Data, colors, labels):
        axs[0].semilogy(Data_["num_grads"], Data_["mean"], color=color, label=label)
        axs[0].fill_between(Data_["num_grads"], Data_["low"], Data_["high"], facecolor=color, alpha=0.3)

    opt_gap_string = r"$\frac{F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi}^*)}{F(\boldsymbol{\xi}_k) - F(\boldsymbol{\xi_0})}$"
    axs[0].set_ylabel(opt_gap_string)
    axs[0].set_xlabel(r"$\mathcal{C}$")
    axs[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axs[0].set_title(f"Logistic regression - {dataset} dataset, interval: {interval * 100:.0f}%")
    axs[0].legend()

    for Data_, color in zip(Data, colors):
        axs[1] = plot_mice(Data_, axs[1], x="iteration", y="mean", markers=False, color=color, style="semilogy")
        axs[1].fill_between(Data_["iteration"], Data_["low"], Data_["high"], facecolor=color, alpha=0.3)
    axs[1].set_ylabel(opt_gap_string)
    axs[1].set_xlabel("Iteration")
    axs[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    for Data_, color in zip(Data, colors):
        axs[2] = plot_mice(Data_, axs[2], x="runtime", y="mean", markers=False, color=color, style="semilogy")
        axs[2].fill_between(Data_["runtime"], Data_["low"], Data_["high"], facecolor=color, alpha=0.3)
    axs[2].set_ylabel(opt_gap_string)
    axs[2].set_xlabel("Runtime (s)")

    for ax in axs:
        for ln in ax.lines:
            ln.set_marker("o")
            ln.set_ms(1)

    fig.tight_layout()
    target_dir = out_dir if out_dir is not None else base_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(target_dir / f"relative_loss_gap_{dataset}_stat.pdf"))


def _parse_seeds(spec: str) -> np.ndarray:
    """Parse seeds: comma list (0,1,2) or inclusive range (0:99)."""
    spec = spec.strip()
    if ":" in spec and "," not in spec:
        a, b = spec.split(":", 1)
        start, end = int(a), int(b)
        if end < start:
            raise ValueError("seed range must be start <= end")
        return np.arange(start, end + 1)
    return np.array([int(s) for s in spec.split(",") if s.strip() != ""])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--seeds", type=str, default="0", help="Comma list (0,1,2) or inclusive range (0:99)")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--outdir", type=str, default=None, help="Optional output directory for generated PDF(s)")
    args = p.parse_args()

    base_dir = Path(__file__).resolve().parent
    out_dir = Path(args.outdir).resolve() if args.outdir else None
    seeds = _parse_seeds(args.seeds)

    if args.stats:
        plots_losses_statistics(args.dataset, seeds, base_dir=base_dir, out_dir=out_dir)
    else:
        plots_losses(args.dataset, base_dir=base_dir, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
