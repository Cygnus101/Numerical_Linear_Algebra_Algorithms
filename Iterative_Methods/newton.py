import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import factorized
from scipy.linalg import cho_factor, cho_solve
from common import iteration_info


def newton(A, b, x0=None, tol=1e-8, maxiter=5):
    x = np.zeros_like(b, dtype=np.float64) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    if issparse(A):
        solve = factorized(A.astype(np.float64).tocsc())
    else:
        C = cho_factor(np.asarray(A, dtype=np.float64))
        solve = lambda r: cho_solve(C, r)
    k = 0
    target = tol * max(np.linalg.norm(b), np.finfo(float).tiny)
    while k < maxiter:
        r = np.asarray(b - A @ x)
        if np.linalg.norm(r) <= target:
            break
        x += solve(r)
        k += 1
    return x, iteration_info(A, x, b, k, tol)
