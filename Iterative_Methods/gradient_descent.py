import numpy as np
from common import iteration_info


def gradient_descent(A, b, x0=None, tol=1e-8, maxiter=10000):
    x = np.zeros_like(b, dtype=np.float64) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    target = tol * max(np.linalg.norm(b), np.finfo(float).tiny)
    k = 0
    while k < maxiter:
        r = np.asarray(b - A @ x)
        if np.linalg.norm(r) <= target:
            break
        q = A @ r
        d = r @ q
        if d <= 0 or not np.isfinite(d):
            raise ValueError("Gradient descent requires SPD A")
        x += ((r @ r) / d) * r
        k += 1
    return x, iteration_info(A, x, b, k, tol)
