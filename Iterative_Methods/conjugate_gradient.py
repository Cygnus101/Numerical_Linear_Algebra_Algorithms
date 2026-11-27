import numpy as np
from common import iteration_info


def conjugate_gradient(A, b, x0=None, tol=1e-8, maxiter=10000):
    x = np.zeros_like(b, dtype=np.float64) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    r = np.asarray(b - A @ x)
    p = r.copy()
    rr = r @ r
    target = tol * max(np.linalg.norm(b), np.finfo(float).tiny)
    k = 0
    while np.sqrt(rr) > target and k < maxiter:
        q = A @ p
        d = p @ q
        if d <= 0 or not np.isfinite(d):
            raise ValueError("CG requires positive curvature (SPD A)")
        a = rr / d
        x += a * p
        r -= a * q
        k += 1
        if np.linalg.norm(r) <= target:
            r = np.asarray(b - A @ x)
            new = r @ r
            if new <= target**2:
                rr = new
                break
            p = r.copy()
        else:
            new = r @ r
            p = r + (new / rr) * p
        rr = new
    return x, iteration_info(A, x, b, k, tol)
