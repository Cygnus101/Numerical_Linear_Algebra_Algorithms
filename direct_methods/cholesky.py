import numpy as np
from numba import njit


@njit
def cholesky(A):
    n = len(A)
    if not np.allclose(A, A.T):
        raise ValueError("A must be symmetric positive definite")
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1):
            s = A[i, j] - np.dot(L[i, :j], L[j, :j])
            if i == j:
                if s <= 0:
                    raise ValueError("A must be positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L
