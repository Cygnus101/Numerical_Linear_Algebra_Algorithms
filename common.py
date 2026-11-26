import numpy as np

def loader(mode = "input", n = "10"):
    if mode == "input":
        raw = input()  # Enter: [[1 2 3],[4 5 6]]

        rows = raw.strip()[2:-2].split("],[")
        matrix = np.array([[int(x) for x in row.split()] for row in rows])

    elif mode == "generate":

        n = int(input)
        rng = np.random.default_rng(seed=42)
        matrix = np.arange(n * n).reshape(n, n)
    
    return matrix