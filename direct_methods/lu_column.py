import numpy as np
from numba import njit


@njit
def lu_column(A):
    C = A.astype(np.float64).copy()
    n = len(C)
    for k in range(n):
        if C[k, k] == 0:
            raise ValueError("Zero pivot; use partial pivoting")
        for i in range(k+1, n):
            C[i, k] /= C[k, k]
        for j in range(k+1, n):
            for i in range(k+1, n):
                C[i, j] -= C[i, k] * C[k, j]
    return np.tril(C, -1) + np.eye(n), np.triu(C)
