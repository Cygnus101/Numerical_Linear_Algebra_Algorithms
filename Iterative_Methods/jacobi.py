import numpy as np
from common import iteration_info


def jacobi(A, b, x0=None, tol=1e-8, maxiter=10000):
    x = np.zeros_like(b, dtype=np.float64) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    d = A.diagonal().astype(np.float64)
    if np.any(d == 0):
        raise ValueError("Zero diagonal")
    target = tol * max(np.linalg.norm(b), np.finfo(float).tiny)
    k = 0
    with np.errstate(over="ignore", invalid="ignore"):
        while k < maxiter:
            r = np.asarray(b - A @ x)
            if np.linalg.norm(r) <= target or not np.all(np.isfinite(r)):
                break
            x += r / d
            k += 1
    return x, iteration_info(A, x, b, k, tol)
