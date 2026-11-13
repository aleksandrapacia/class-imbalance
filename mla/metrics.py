import numpy as np
import autograd.numpy as anp


def mean_squared_error(y_true, y_pred):
    return anp.mean((y_true - y_pred) ** 2)


def binary_crossentropy(y, p):
    p = anp.clip(p, 1e-12, 1.0 - 1e-12)
    return -anp.mean(y * anp.log(p) + (1.0 - y) * anp.log(1.0 - p))


def accuracy(y, p):
    return np.mean(y == p)


def focal_loss(y, p, gamma, alpha_pos, alpha_neg):
    """Focal loss with a separate alpha for each class.

    alpha_pos weights the positive class. alpha_neg weights the negative class.
    gamma down-weights examples the model already classifies confidently.
    """
    p = anp.clip(p, 1e-12, 1.0 - 1e-12)
    loss_pos = y * (-alpha_pos) * (1.0 - p) ** gamma * anp.log(p)
    loss_neg = (1.0 - y) * (-alpha_neg) * p ** gamma * anp.log(1.0 - p)
    return anp.mean(loss_pos + loss_neg)


def compute_alphas(y):
    n_total = len(y)
    n_pos = y.sum()
    n_neg = n_total - n_pos
    if n_pos <= n_neg:
        alpha_pos = n_neg / n_total
        alpha_neg = n_pos / n_total
    else:
        alpha_pos = n_pos / n_total
        alpha_neg = n_neg / n_total
    return alpha_pos, alpha_neg
