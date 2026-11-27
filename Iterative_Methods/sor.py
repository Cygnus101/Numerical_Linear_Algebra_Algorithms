import numpy as np
from scipy.sparse import csr_matrix
from numba import njit
from common import iteration_info


@njit
def _sweep(data, indices, indptr, d, b, x, omega):
    for i in range(len(b)):
        s = 0.0
        for j in range(indptr[i], indptr[i+1]):
            s += data[j] * x[indices[j]]
        x[i] += omega * (b[i] - s) / d[i]


def sor(A, b, omega=1.1, x0=None, tol=1e-8, maxiter=10000):
    omega = omega
    if not 0 < omega < 2:
        raise ValueError("Require 0 < omega < 2")
    C = csr_matrix(A, dtype=np.float64)
    C.sum_duplicates()
    d = C.diagonal()
    if np.any(d == 0):
        raise ValueError("Zero diagonal")
    x = np.zeros_like(b, dtype=np.float64) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    k = 0
    while k < maxiter:
        info = iteration_info(C, x, b, k, tol)
        if info["converged"] or not np.isfinite(info["residual"]):
            break
        _sweep(C.data, C.indices, C.indptr, d, b, x, omega)
        k += 1
    return x, iteration_info(C, x, b, k, tol)
