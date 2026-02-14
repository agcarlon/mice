from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit, softplus

from .configs import get_config
from .datasets import ensure_dataset_npy


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return expit(z)


def compute_optimum_lbfgs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    Lambda: float,
    x0: np.ndarray,
    max_iter: int,
    gtol: float,
):
    try:
        from scipy.optimize import minimize
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("SciPy is required to compute Optimum.npy via L-BFGS-B") from e

    n = float(X.shape[0])

    def f(w: np.ndarray) -> float:
        # Paper objective: log(1 + exp(-y <w,x>)) + (lambda/2)||w||^2; softplus is stable for large |z|
        z = -y * (X @ w)
        return float(np.mean(softplus(z)) + 0.5 * Lambda * (w @ w))

    def g(w: np.ndarray) -> np.ndarray:
        z = -y * (X @ w)
        # grad = mean( -y * x * sigmoid(-y <w,x>) ) + Lambda*w
        s = _sigmoid(z) * (-y)
        return (s @ X) / n + Lambda * w

    res = minimize(
        fun=f,
        x0=x0,
        jac=g,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "gtol": float(gtol)},
    )
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mushrooms")
    p.add_argument("--lambda", dest="Lambda", type=float, default=None, help="Override dataset lambda from configs.py.")
    p.add_argument("--via", type=str, default="lbfgs", choices=("lbfgs", "sgd_mice"),
                   help="How to compute the optimum: 'lbfgs' (L-BFGS-B) or 'sgd_mice' (SGD-MICE with larger budget and fixed seed).")
    p.add_argument("--epochs", type=int, default=None, help="Override dataset epochs from configs.py (used by --via sgd_mice).")
    p.add_argument("--seed", type=int, default=1000, help="Seed when --via sgd_mice (default 1000; test seeds are 0-99).")
    p.add_argument("--eps", type=float, default=None, help="Step-size when --via sgd_mice (default 1/sqrt(3) to match run_all).")
    p.add_argument("--max-iter", type=int, default=2000, help="L-BFGS-B max iterations (--via lbfgs; increase for large datasets, e.g. 5000 for Higgs)")
    p.add_argument("--gtol", type=float, default=1e-10)
    p.add_argument("--max-samples", type=int, default=0, help="If > 0, use only the first N samples (L-BFGS only; for a faster approximate optimum on huge data, e.g. Higgs)")
    p.add_argument("--outdir", type=str, default="")
    args = p.parse_args()
    config = get_config(args.dataset)
    lambda_reg = float(args.Lambda) if args.Lambda is not None else float(config["lambda_reg"])
    epochs = int(args.epochs) if args.epochs is not None else int(config["epochs"])

    base_dir = Path(__file__).resolve().parent
    ds_dir = Path(args.outdir).resolve() if args.outdir else (base_dir / args.dataset)
    ds_dir.mkdir(parents=True, exist_ok=True)

    npy_path = ds_dir / f"{args.dataset}.npy"
    if not npy_path.exists():
        print(f"Missing dataset npy at {npy_path}. Preparing dataset automatically...")
        prepared = ensure_dataset_npy(args.dataset, base_dir=base_dir)
        npy_path = prepared
        print(f"Using prepared dataset: {npy_path}")

    if args.via == "sgd_mice":
        try:
            from .run_logreg import sgd_mice
        except Exception as e:
            raise RuntimeError("SGD-MICE optimum requires the mice package on PYTHONPATH (see run_all)") from e
        eps = float(args.eps) if args.eps is not None else 1.0 / (3.0 ** 0.5)
        w_opt, loss_final = sgd_mice(
            args.dataset, eps, lambda_reg, epochs,
            seed=args.seed, for_optimum=True,
        )
        w_opt = np.asarray(w_opt, dtype=float)
        out_path = ds_dir / "Optimum.npy"
        np.save(str(out_path), w_opt)
        print(f"saved: {out_path} (SGD-MICE {epochs} epochs, seed={args.seed}, final_loss={loss_final:.6f})")
        return 0

    thetas = np.load(str(npy_path), allow_pickle=True)
    X = np.stack(thetas[:, 0]).astype(float, copy=False)
    y = np.asarray(thetas[:, 1], dtype=float)
    n_features = X.shape[1]

    if getattr(args, "max_samples", 0) and int(args.max_samples) > 0:
        n = int(args.max_samples)
        if n < X.shape[0]:
            X = X[:n].copy()
            y = y[:n].copy()
            print(f"Using first {n} samples (approximate optimum)")

    x0 = np.zeros(n_features, dtype=float)
    res = compute_optimum_lbfgs(X, y, Lambda=lambda_reg, x0=x0, max_iter=args.max_iter, gtol=args.gtol)
    w_opt = np.asarray(res.x, dtype=float)
    final_loss = float(res.fun) if getattr(res, "fun", None) is not None else float("nan")

    out_path = ds_dir / "Optimum.npy"
    np.save(str(out_path), w_opt)

    print(f"saved: {out_path}")
    print(f"final_loss={final_loss:.6f}")
    print(f"success={res.success} status={res.status} nit={res.nit} nfev={res.nfev}")
    print(f"final_grad_norm={float(np.linalg.norm(res.jac)):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
