"""
Shared data loading and Lipschitz constants for fair comparison across methods.

- load_dataset_shuffled(): same data order for a given seed (all methods).
- SGD-MICE uses L from the *mean* (smoothness of the true objective F = E[f(·,θ)]):
  compute_L_hessian() = max eigenvalue of E[H(θ)] = max eig(0.25*X'X/n + Lambda*I).
- SAG/SAGA/SVRG/SARAH use L̂ that holds *a.s. for all data points* (per the paper):
  compute_L_hat() = 0.25*max_i ||x_i||^2 + Lambda.
See submission_4/main.tex §logistic regression and Table step-sizes (SGD-MICE vs L̂).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_dataset_shuffled(data_path: Path | str, seed: int):
    """
    Load dataset from .npy (array of [x, y] rows) and shuffle with seed.
    Returns (X, Y, thetas_list, thetas_mat, datasize, n_features) so all methods
    see the same sample order for a given seed.
    """
    thetas = np.load(str(data_path), allow_pickle=True)
    X = np.stack(thetas[:, 0]).astype(float, copy=False)
    Y = np.asarray(thetas[:, 1], dtype=float)
    datasize, n_features = X.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(datasize)
    X = X[perm]
    Y = Y[perm]
    thetas_list = [*zip(X, Y)]  # for SAG/SAGA/SVRG/SARAH
    thetas_mat = np.concatenate([Y.reshape(-1, 1), X], axis=1).astype(float, copy=False)  # for MICE
    return X, Y, thetas_list, thetas_mat, datasize, n_features


def compute_L_hessian(X: np.ndarray, datasize: int, Lambda: float) -> float:
    """L from mean Hessian E[H(θ)]: max eigenvalue of 0.25*X'X/n + Lambda*I. For SGD-MICE (Proposition, true objective)."""
    from scipy.sparse.linalg import eigsh as largest_eigsh
    hess = 0.25 * (X.T @ X) / datasize + Lambda * np.eye(X.shape[1])
    return float(largest_eigsh(hess, 1, which="LM")[0][0])


def compute_L_hat(X: np.ndarray, Lambda: float) -> float:
    """L̂ that holds a.s. for all data points: 0.25*max_i ||x_i||^2 + Lambda. For SAG/SAGA/SVRG/SARAH (paper Table)."""
    return float(0.25 * np.max((X**2).sum(axis=1)) + Lambda)


def load_data(filename, datasize, n_features):
    # Keep this for parity with v1 scripts (LIBSVM-like sparse text format).
    X = np.zeros((datasize, n_features))
    Y = np.zeros((datasize,))

    with open(filename) as f:
        for i, line in enumerate(f):
            if i == datasize:
                break
            data = line.split()
            Y[i] = int(data[0])  # target value
            for item in data[1:]:
                j, k = item.split(":")
                X[i, int(j) - 1] = float(k)
    return X, Y


def normalize(X):
    norms = np.linalg.norm(X, axis=1)
    X_ = np.array([x / (norm + 1e-20) for x, norm in zip(X, norms)])
    return X_

