import numpy as np
import autograd.numpy as anp
from autograd import grad


class BaseEstimator:
    def _setup_input(self, X, y=None):
        self.X = np.array(X, dtype=float)
        if y is not None:
            self.y = np.array(y, dtype=float)

    def predict(self, X):
        return self._predict(np.array(X, dtype=float))


class BasicRegression(BaseEstimator):
    def __init__(self, lr=0.001, penalty="None", C=0.01, tolerance=0.0001, max_iters=1000):
        self.C = C
        self.penalty = penalty
        self.tolerance = tolerance
        self.lr = lr
        self.max_iters = max_iters
        self.errors = []
        self.theta = []
        self.n_samples = None
        self.n_features = None
        self.cost_func = None

    def _loss(self, w):
        raise NotImplementedError()

    def init_cost(self):
        raise NotImplementedError()

    def _add_penalty(self, loss, w):
        if self.penalty == "l1":
            loss += self.C * anp.abs(w[1:]).sum()
        elif self.penalty == "l2":
            loss += (0.5 * self.C) * (w[1:] ** 2).sum()
        return loss

    def _cost(self, X, y, theta):
        prediction = X.dot(theta)
        return self.cost_func(y, prediction)

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.init_cost()
        self.n_samples, self.n_features = self.X.shape
        self.theta = np.random.normal(size=(self.n_features + 1), scale=0.5)
        self.X = self._add_intercept(self.X)
        self._train()

    @staticmethod
    def _add_intercept(X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def _train(self):
        self.theta, self.errors = self._gradient_descent()

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return X.dot(self.theta)

    def _gradient_descent(self):
        theta = self.theta.copy()
        errors = [self._cost(self.X, self.y, theta)]
        cost_d = grad(self._loss)
        for i in range(1, self.max_iters + 1):
            delta = cost_d(theta)
            theta -= self.lr * delta
            errors.append(self._cost(self.X, self.y, theta))
            error_diff = np.linalg.norm(errors[i - 1] - errors[i])
            if error_diff < self.tolerance:
                break
        return theta, errors
