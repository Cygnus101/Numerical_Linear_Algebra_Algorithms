import numpy as np


def power_method(A, x0=None, tol=1e-8, maxiter=10000):
    n = A.shape[0]
    x = np.random.default_rng(42).normal(size=n) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    x /= np.linalg.norm(x)
    for k in range(maxiter+1):
        y = A @ x
        lam = float(x @ y)
        scale = max(np.linalg.norm(y), np.finfo(float).tiny)
        r = float(np.linalg.norm(y - lam*x) / scale)
        if r <= tol or k == maxiter:
            return lam, x, {"iterations": k, "residual": r, "converged": r <= tol}
        if not np.isfinite(scale):
            raise ValueError("Nonfinite iterate")
        x = y / scale
