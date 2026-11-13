import autograd.numpy as anp

from mla.base import BasicRegression
from mla.metrics import binary_crossentropy, compute_alphas, focal_loss, mean_squared_error
import numpy as np


class LinearRegression(BasicRegression):
    """Linear regression trained by gradient descent."""

    def _loss(self, w):
        loss = self.cost_func(self.y, anp.dot(self.X, w))
        return self._add_penalty(loss, w)

    def init_cost(self):
        self.cost_func = mean_squared_error


class LogisticRegression(BasicRegression):
    """Binary logistic regression with unweighted cross-entropy."""

    def init_cost(self):
        self.cost_func = binary_crossentropy

    def _loss(self, w):
        loss = self.cost_func(self.y, self.sigmoid(anp.dot(self.X, w)))
        return self._add_penalty(loss, w)

    @staticmethod
    def sigmoid(x):
        return 0.5 * (anp.tanh(0.5 * x) + 1)

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return self.sigmoid(X.dot(self.theta))


class LogisticRegressionFocal(BasicRegression):
    """Binary logistic regression trained with focal loss."""

    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha_pos = None
        self.alpha_neg = None

    def fit(self, X, y=None):
        y_arr = np.array(y, dtype=float)
        self.alpha_pos, self.alpha_neg = compute_alphas(y_arr)
        super().fit(X, y)

    def init_cost(self):
        self.cost_func = lambda y, p: focal_loss(
            y, p,
            gamma=self.gamma,
            alpha_pos=self.alpha_pos,
            alpha_neg=self.alpha_neg,
        )

    def _loss(self, w):
        loss = self.cost_func(self.y, self.sigmoid(anp.dot(self.X, w)))
        return self._add_penalty(loss, w)

    @staticmethod
    def sigmoid(x):
        return 0.5 * (anp.tanh(0.5 * x) + 1)

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return self.sigmoid(X.dot(self.theta))
