import numpy as np
from math import pi
EPSILON = 1e-6


def nearPSD(A, epsilon=0):
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval,epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec,vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B*B.T
    return(out)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def log_sum(weight_array):
    '''
    weight_sum = log_sum(weight_array)

    Sum of logarithmic components
    w_sum = w_smallest + log( 1 + sum(exp(w_rest - w_smallest)) )
    '''
    weight_array.sort()
    _w0 = weight_array[0]
    _wr = weight_array[1:]
    _wdelta = _wr - _w0
    _exp = np.exp(_wdelta)
    _sum = np.sum(_exp)
    _weight = _w0 + np.log(1 + _sum)

    return _weight


def switch_state_direction(state):
    output_state = np.copy(state)
    if state[3] < 0:
        if output_state[2] > 0:
            output_state[2] -= pi
        else:
            output_state[2] += pi
        output_state[3] = abs(output_state[3])
    return output_state

