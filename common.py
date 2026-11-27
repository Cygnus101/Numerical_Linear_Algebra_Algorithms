import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse

def loader(mode="input", n=10, path=None, dense=True):

    if mode == "input":
        raw = input()
        rows = raw.strip()[2:-2].split("],[")
        matrix = np.array([[float(x) for x in row.split()] for row in rows])

    elif mode == "generate":
        rng = np.random.default_rng(seed=42)
        matrix = rng.integers(1, 10, size=(n, n)).astype(float)

    elif mode == "suitesparse":
        data = loadmat(path)
        matrix = data["Problem"][0, 0]["A"].astype(float)

        if dense and issparse(matrix):
            matrix = matrix.toarray()

    else:
        raise ValueError("Unknown loader mode")

    return matrix


def data_path(name, path=None):
    from pathlib import Path
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    local = Path(__file__).resolve().parent / "data" / f"{name}.mat"
    if local.is_file():
        return local
    matches = sorted(Path("/kaggle/input").rglob(f"{name}.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Attach {name}.mat or set an explicit path; matches: {matches}")
    return matches[0]


def relative_residual(A, x, b):
    return float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), np.finfo(float).tiny))


def backward_substitution(U, b):
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(len(b)-1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Zero pivot")
        x[i] = (b[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x


def lu_solve(L, U, b, p=None):
    c = np.array(b if p is None else b[p], dtype=np.float64, copy=True)
    for i in range(len(c)):
        c[i] = (c[i] - L[i, :i] @ c[:i]) / L[i, i]
    return backward_substitution(U, c)


def iteration_info(A, x, b, k, tol):
    r = relative_residual(A, x, b)
    return {"iterations": k, "residual": r, "converged": bool(np.isfinite(r) and r <= tol)}
