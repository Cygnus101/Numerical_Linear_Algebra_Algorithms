import numpy as np
from numba import njit


@njit
def householder_qr(A):
    R = A.astype(np.float64).copy()
    m, n = R.shape
    V = np.zeros((m, min(m, n)))
    for k in range(min(m, n)):
        v = R[k:, k].copy()
        s = np.linalg.norm(v)
        if s == 0:
            continue
        v[0] += s if v[0] >= 0 else -s
        v /= np.linalg.norm(v)
        V[k:, k] = v
        for j in range(k, n):
            R[k:, j] -= 2 * np.dot(v, R[k:, j]) * v
        R[k+1:, k] = 0
    return V, R


@njit
def apply_q(V, B, transpose=False):
    C = B.astype(np.float64).copy()
    n = V.shape[1]
    for t in range(n):
        k = t if transpose else n-1-t
        v = V[k:, k]
        for j in range(C.shape[1]):
            C[k:, j] -= 2 * np.dot(v, C[k:, j]) * v
    return C
