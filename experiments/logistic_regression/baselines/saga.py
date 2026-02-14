from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from scipy.special import expit, softplus

from ..configs import get_config
from ..datasets import ensure_dataset_npy
from ..load_data import compute_L_hat, load_dataset_shuffled
from .optimizers import SAGA


def sgd_saga(dataset, batchsize, Lambda, epochs, seed=0):
    print(f"Solving logistic regression of {dataset} dataset using SAGA")
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / dataset
    data_path = ensure_dataset_npy(dataset, base_dir=base_dir)
    opt_path = dataset_dir / "Optimum.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_path}")
    if not opt_path.exists():
        raise FileNotFoundError(f"Missing optimum file: {opt_path}")

    out_dir = dataset_dir / "saga" / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_dir / "saga"

    t0 = time()
    X, Y, _, thetas_mat, datasize, n_features = load_dataset_shuffled(data_path, seed)
    X = np.asarray(X, dtype=float, copy=False)
    Y = np.asarray(Y, dtype=float, copy=False)
    print(f"Load time: {time() - t0:.3f}s")

    opt_W = np.load(str(opt_path)).astype(float, copy=False)
    W0 = np.zeros(shape=(n_features))

    t0 = time()
    L = compute_L_hat(X, Lambda)  # L̂: holds a.s. for all data points (paper)
    print(f"L: {L:.6g}")
    print(f"L estim. time: {time() - t0:.3f}s")

    def logloss_full(W):
        z = -Y * (X @ W)
        return float(np.mean(softplus(z)) + 0.5 * Lambda * (W @ W))

    def lossgrad(W: np.ndarray, batch: np.ndarray) -> np.ndarray:
        # batch: (m, 1+d), y in col 0, x in 1:
        yb = batch[:, 0]
        xb = batch[:, 1:]
        z = -yb * (xb @ W)
        return (expit(z) * (-yb))[:, None] * xb

    opt_loss = logloss_full(opt_W)
    start_loss = logloss_full(W0)
    print(f"start loss: {start_loss:.6f}, opt loss:{opt_loss:.6f}")

    dF = SAGA(lossgrad, sample=thetas_mat, max_cost=epochs * datasize, batchsize=batchsize, verbose=False)

    stepsize = 1 / (16 * (L + Lambda * datasize)) if n_features < datasize else 1 / L
    k = 1
    W = [W0]

    epoch = 0
    times_epoch = [1e-2]
    losses = [start_loss]
    epochs_ = [0]
    kepoch = [1]
    counter = [1]

    t0 = time()
    np.random.seed(0)
    while not dF.force_exit:
        k += 1
        grad = dF.evaluate(W[-1]) + Lambda * W[-1]
        if dF.force_exit:
            break
        W.append(W[-1] - stepsize * grad)
        if dF.counter >= (epoch + 1) * datasize:
            times_epoch.append(time() - t0)
            loss_k = logloss_full(W[-1])
            losses.append(loss_k)
            counter.append(dF.counter)
            epoch = int(np.floor(dF.counter / datasize))
            epochs_.append(epoch)
            kepoch.append(k)
            t0 = time()

    times_epoch.append(time() - t0)
    runtime = np.array(times_epoch)
    loss_k = logloss_full(W[-1])
    losses.append(loss_k)
    counter.append(dF.counter)
    epoch = int(np.floor(dF.counter / datasize))
    epochs_.append(epoch)
    kepoch.append(k)

    data = np.array([epochs_, losses, counter, kepoch, runtime])
    log = pd.DataFrame(data.T, columns=["epochs", "losses", "num_grads", "iteration", "runtime"])
    log["rel_loss"] = (log["losses"] - opt_loss) / (start_loss - opt_loss)
    log.to_pickle(str(name) + ".pkl")
    print(f"Relative loss opt: {float(log['rel_loss'].dropna().iloc[-1]):.3e}")
    print(f"runtime(total): {float(np.sum(runtime)):.3f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--batchsize", type=int, default=10)
    p.add_argument("--lambda", dest="Lambda", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    config = get_config(args.dataset)
    lambda_reg = float(args.Lambda) if args.Lambda is not None else float(config["lambda_reg"])
    epochs = int(args.epochs) if args.epochs is not None else int(config["epochs"])
    sgd_saga(args.dataset, args.batchsize, lambda_reg, epochs, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
