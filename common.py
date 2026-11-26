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

    return matrix