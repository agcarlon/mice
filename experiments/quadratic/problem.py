from __future__ import annotations

import numpy as np


def make_problem(*, kappa: float = 100.0, seed: int = 1, dim: int = 2):
    """
    Quadratic test problem with controllable condition number and dimension.

    For dim == 2, this reproduces the original setup used in `mice_numerics/quad_c.py`.
    For dim > 2, we construct a diagonal H1 so that the mean Hessian
    EH = 0.5 * (H0 + H1) has condition number approximately kappa.

    Returns (dobjf, Eobjf, Edobjf, sampler, optimum, f_opt, L, stepsize).
    """
    np.random.seed(seed)

    if dim == 2:
        # Original 2D setup (kept for backward compatibility with existing numerics)
        H0 = np.eye(2)
        H1 = np.array([[kappa * 2.0, 0.5], [0.5, 1.0]])
        b = np.ones(2)
    else:
        # General d-dimensional setup.
        # H0 is identity; H1 is diagonal with a larger first eigenvalue so that
        # EH = 0.5 * (H0 + H1) has spectrum {kappa, 1, ..., 1} and thus condition
        # number kappa in the dominant directions.
        H0 = np.eye(dim)
        diag = np.ones(dim)
        # Choose H1 such that the first eigenvalue of EH is exactly kappa:
        # (1 + lambda_1(H1)) / 2 = kappa  =>  lambda_1(H1) = 2*kappa - 1.
        diag[0] = 2.0 * kappa - 1.0
        H1 = np.diag(diag)
        b = np.ones(dim)

    EH = 0.5 * (H0 + H1)

    def objf(x: np.ndarray, theta: float) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        H = H0 * (1.0 - theta) + H1 * theta
        return float(0.5 * (x @ H @ x) - b @ x)

    def dobjf(x: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """
        Stochastic gradient of the quadratic objective.

        For dim == 2 this matches the original `quad_c.py` behavior.
        For general dim we use the identity:
          grad(x; theta) = H(theta) x - b,
          H(theta) = (1-theta) H0 + theta H1.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        thetas = np.asarray(thetas, dtype=float).reshape(-1)

        if dim == 2:
            # Original 2D expression (kept for exact parity with earlier experiments)
            grad = np.outer(x, (1.0 - thetas)).T + np.outer((x @ H1), thetas).T - b
            return grad

        # General d-dimensional case.
        # Precompute H0 x and H1 x, then interpolate for each theta.
        H0x = H0 @ x
        H1x = H1 @ x
        # Each row i: (1 - theta_i) * H0x + theta_i * H1x - b
        grad = np.outer(1.0 - thetas, H0x) + np.outer(thetas, H1x) - b
        return grad

    def Eobjf(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        return float(0.5 * (x @ EH @ x) - b @ x)

    def Edobjf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        return x @ EH - b

    def sampler(n: int) -> np.ndarray:
        return np.random.uniform(0.0, 1.0, int(n))

    optimum = np.linalg.solve(EH, b)
    f_opt = Eobjf(optimum)
    eigs = np.linalg.eig(EH)[0]
    L = float(eigs.max())
    stepsize = 1.0 / L

    return dobjf, objf, Eobjf, Edobjf, sampler, optimum, f_opt, L, stepsize

