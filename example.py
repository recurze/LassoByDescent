import time
import numpy as np
import scipy as sp

from lasso import (
    DescentLassoCD,
    DescentLassoPG,
    DescentLassoAPG,
    DescentLassoAPGRestart
)
from sklearn.preprocessing import StandardScaler


def generate_XY(n, p):
    X = StandardScaler().fit_transform(np.random.normal(size=(n, p)))

    bstar = sp.sparse.random(p, 1, density=0.05, random_state=0).toarray()
    epsilon = np.random.normal(size=(n, 1))
    Y = (X @ bstar) + 0.01 * epsilon

    assert X.shape == (n, p)
    assert Y.shape == (n, 1)
    assert bstar.shape == (p, 1)

    return X, Y, bstar


if __name__ == "__main__":
    def objective(X, Y, lam, beta):
        f = (np.linalg.norm((X @ beta) - Y)**2) / 2
        g = lam * np.linalg.norm(beta, ord=1)
        return f, g, f + g

    n, p = 1000, 2500
    print(f"Generating XY {n}x{p}")
    X, Y, bstar = generate_XY(n, p)

    print("Precomputing XtX and XtY")
    xtx = X.T @ X
    xty = X.T @ Y

    assert xtx.shape == (p, p)
    assert xty.shape == (p, 1)

    print("Computing parameters alpha and lambda")
    start = time.time()
    alpha = 1/np.linalg.eigvalsh(xtx).max()
    lam = 0.1 * np.linalg.norm(xty, ord=np.inf)
    setup_time = time.time() - start
    print(f"Setup time: {setup_time}\n")

    print("\nStayc girls, it's going down!")
    print("-----------------------------------------------------------\n")

    args = (xtx, xty, alpha, lam)
    for descentLasso in [
        DescentLassoCD(xtx, xty, 1/n, lam),
        DescentLassoPG(*args),
        DescentLassoAPG(*args),
        DescentLassoAPGRestart(100, *args),
    ]:
        start = time.time()
        descentLasso.descent()
        solve_time = time.time() - start

        print(f"Method: {descentLasso.__class__}\n"
              f"Solve time: {solve_time}\n")

        print("Objective (f, g, f + g):")
        print(f"bstar      : {objective(X, Y, lam, bstar)}")
        print(f"Initial (0): {objective(X, Y, lam, np.zeros(shape=(p, 1)))}")
        print(f"Final (b_{descentLasso.k}): "
              f"{objective(X, Y, lam, descentLasso.b)}")
        print("-----------------------------------------------------------\n")
