import numpy as np
from scipy.sparse import issparse, eye
from scipy.sparse.linalg import factorized
from scipy.linalg import lu_factor, lu_solve


def inverse_power_method(A, shift=0.0, x0=None, tol=1e-8, maxiter=10000):
    n = A.shape[0]
    if issparse(A):
        solve = factorized((A.astype(np.float64) - shift*eye(n)).tocsc())
    else:
        C = lu_factor(np.asarray(A, dtype=np.float64) - shift*np.eye(n))
        solve = lambda r: lu_solve(C, r)
    x = np.random.default_rng(42).normal(size=n) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    x /= np.linalg.norm(x)
    for k in range(maxiter+1):
        y = A @ x
        lam = float(x @ y)
        r = float(np.linalg.norm(y - lam*x) / max(np.linalg.norm(y), np.finfo(float).tiny))
        if r <= tol or k == maxiter:
            return lam, x, {"iterations": k, "residual": r, "converged": r <= tol}
        y = solve(x)
        if not np.all(np.isfinite(y)):
            raise ValueError("Singular shifted matrix")
        x = y / np.linalg.norm(y)
