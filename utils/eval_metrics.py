import numpy as np
import scipy.optimize as spopt
from scipy.spatial.distance import cdist


# "Generalized optimal sub-pattern assignment metric"
# Rahmathullah et al 2017
def GOSPA(ground_truths, estimates, p=1, c=100, alpha=2., state_dim=2):
    assert ground_truths.shape[1] == state_dim, 'Shape = {}, state_dim = {}'.format(ground_truths.shape[1], state_dim)
    assert estimates.shape[1] == state_dim

    m = len(ground_truths)
    n = len(estimates)
    if m > n:
        return GOSPA(estimates, ground_truths, p, c, alpha, state_dim)
    if m == 0:
        return c ** p / alpha * n
    costs = cdist(ground_truths, estimates)
    costs = np.minimum(costs, c) ** p
    row_ind, col_ind = spopt.linear_sum_assignment(costs)
    return np.sum(costs[row_ind, col_ind]) + c ** p / alpha * (n - m)


# did this myself - like RMS over time
def normalizeGOSPA(ospas, ns, smoothval, p=1):
    ospas = np.convolve(ospas, [1] * smoothval, 'valid')[::smoothval]
    counts = np.convolve(ns, [1] * smoothval, 'valid')[::smoothval]
    return (ospas / counts) ** (1. / p)


def isPSD(matrix): return np.sign(np.linalg.eigvals(matrix))


def isPosDef(matrix): return np.all(np.real(np.linalg.eigvals(matrix)) > 0)


def normpdf(x, prec):
    dev = prec.dot(x).dot(x)

    return (np.linalg.det(prec) / np.exp(dev)) ** .5 / (2 * np.pi) ** 1.5
