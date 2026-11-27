import numpy as np
from numba import njit


@njit
def gram_schmidt(A):
    C = A.astype(np.float64)
    m, n = C.shape
    if m < n:
        raise ValueError("Require m >= n")
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = C[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], C[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] <= np.finfo(np.float64).eps * np.linalg.norm(C[:, j]):
            raise ValueError("Numerically dependent columns")
        if R[j, j] == 0:
            raise ValueError("Dependent columns")
        Q[:, j] = v / R[j, j]
    return Q, R
