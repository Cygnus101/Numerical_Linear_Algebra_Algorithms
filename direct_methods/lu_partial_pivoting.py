import numpy as np
from numba import njit


@njit
def lu_partial_pivoting(A):
    C = A.astype(np.float64).copy()
    n = len(C)
    p = np.arange(n)
    for k in range(n):
        r = k + np.argmax(np.abs(C[k:, k]))
        if C[r, k] == 0:
            raise ValueError("Singular matrix")
        if r != k:
            for j in range(n):
                C[k, j], C[r, j] = C[r, j], C[k, j]
            p[k], p[r] = p[r], p[k]
        for i in range(k+1, n):
            C[i, k] /= C[k, k]
            for j in range(k+1, n):
                C[i, j] -= C[i, k] * C[k, j]
    return np.tril(C, -1) + np.eye(n), np.triu(C), p
