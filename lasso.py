import functools
import numpy as np
from descent import Descent


class DescentLasso(Descent):
    def __init__(self, xtx, xty, alpha, lam, *args):
        self.xtx = xtx
        self.xty = xty
        self.alpha = alpha
        self.lam = lam
        super().__init__(
            functools.partial(DescentLasso._compute_delta, self.xtx),
            np.zeros_like(self.xty),
            -self.xty,
            *args
        )

    @staticmethod
    def matrix_dot_sparse_vector(A, v):
        nonzero_rows = v.nonzero()[0]
        return A[:, nonzero_rows] @ v[nonzero_rows]

    @staticmethod
    def _compute_delta(xtx, delta, b_diff):
        # Computing delta is the costliest fn (dot product)
        # Keep a running sum to get rid of subtraction (xty)
        delta += DescentLasso.matrix_dot_sparse_vector(xtx, b_diff)

    @staticmethod
    def soft_threshold(lam, v):
        return np.sign(v) * np.maximum(np.abs(v) - lam, np.zeros_like(v))

    def get_error(self):
        return np.linalg.norm(
            self.b - DescentLasso.soft_threshold(self.lam, self.b - self.delta)
        )

    def take_step(self):
        raise "Not implemented"


class DescentLassoPG(DescentLasso):
    def take_step(self):
        self.b_prev = self.b
        self.b = DescentLasso.soft_threshold(self.alpha * self.lam,
                                             self.b - self.alpha * self.delta)
        self.k += 1


class DescentLassoAPG(DescentLasso):
    def __init__(self, *args):
        self.t = self.t_next = 1
        super().__init__(*args)

    @staticmethod
    def next_t(t):
        return (1 + (1 + 4*t*t)**0.5)/2

    def take_step(self):
        coef = (self.t - 1)/self.t_next
        b_diff = self.b - self.b_prev

        b_dash = self.b + (coef * b_diff)
        delta_bdash = self.delta + coef * (self.xtx @ b_diff)

        self.b_prev = self.b
        self.b = DescentLasso.soft_threshold(self.alpha * self.lam,
                                             b_dash - self.alpha * delta_bdash)
        self.k += 1

        self.t, self.t_next = self.t_next, self.next_t(self.t_next)


class DescentLassoAPGRestart(DescentLassoAPG):
    def __init__(self, restart_every_k_iter=100, *args):
        self.restart_every_k_iter = restart_every_k_iter
        super().__init__(*args)

    def take_step(self):
        if self.k % self.restart_every_k_iter == 0:
            self.t, self.t_next = 1, 1
        super().take_step()


class DescentLassoCD(DescentLasso):
    @staticmethod
    def soft_threshold_single(lam, x):
        return (1 if x > 0 else -1) * max(abs(x) - lam, 0)

    def take_step(self):
        self.b_prev = self.b.copy()
        for i in range(self.b.shape[0]):
            y = self.xtx[i, :].dot(self.b)[0] - self.xty[i, 0]
            self.b[i, 0] = self.soft_threshold_single(
                self.alpha * self.lam,
                self.b[i, 0] - self.alpha * y
            )
        self.k += 1
