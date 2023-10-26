import numpy as np


class Descent:
    def __init__(self,
                 compute_delta,
                 b_0,
                 delta_0,
                 tol=1e-3):
        self.compute_delta = compute_delta
        self.b, self.k, self.b_prev = b_0, 0, b_0
        self.delta = delta_0
        self.tol = tol

    def get_error(self):
        raise "Not implemented"

    def take_step(self):
        raise "Not implemented"

    def descent(self):
        while True:
            self.compute_delta(self.delta, self.b - self.b_prev)

            err = self.get_error()
            if err < self.tol:
                break

            self.take_step()


class GradientDescent(Descent):
    def get_alpha(self, b):
        raise "Not implemented"

    def get_error(self):
        return np.linalg.norm(self.delta)

    def take_step(self):
        self.b_prev = self.b

        alpha = self.get_alpha(self.b)

        self.b = self.b - alpha * self.delta
        self.k += 1


class GradientDescentConstantStep(GradientDescent):
    def __init__(self, alpha, *args):
        super().__init__(*args)
        self.alpha = alpha

    def get_alpha(self, b):
        return self.alpha


class GradientDescentExactLineSearch(GradientDescent):
    def __init__(self, get_alpha, *args):
        super().__init__(*args)
        self.get_alpha = get_alpha
