import numpy as np
from itertools import product


def get_model(model_name):
    if model_name == 'noob':
        _states = [np.array([[-45], [-45], [10], [5]]),
                  np.array([[-45], [45], [10], [-5]]),
                  np.array([[-45], [-45], [2.5], [2.5]]),
                  np.array([[-45], [45], [5], [-5]]),
                  np.array([[-45], [-45], [15], [10]]),
                  np.array([[-45], [45], [7], [-7]]),
                  np.array([[-45], [-45], [10], [5]]),
                  np.array([[-45], [-25], [10], [5]]),
                  np.array([[-38], [-22], [5], [5]]),
                  np.array([[-40], [-25], [1], [1]]),
                  np.array([[-45], [45], [15], [-10]])]
        _variance = 1 * np.eye(4)
        return _states, _variance

    elif model_name == 'noob2':
        _states = [np.array([[-45], [45]]),
                   np.array([[-40], [35]]),
                   np.array([[-42], [40]]),
                   np.array([[-45], [-30]]),
                   np.array([[-45], [-35]]),
                   np.array([[-45], [-45]]),
                   np.array([[-30], [30]]),
                   np.array([[-20], [20]]),
                   np.array([[-35], [35]]),
                   np.array([[-30], [-30]]),
                   np.array([[-20], [-20]]),
                   np.array([[-35], [-35]])]
        _variance = np.eye(2)
        return _states, _variance

    elif model_name == 'datadriven':
        r_one = [np.array([[3], [y], [0], [0]]) for y in range(5, 80, 5)]
        r_two = [np.array([[8], [y], [0], [0]]) for y in range(5, 70, 5)]
        r_three = [np.array([[13], [y], [0], [0]]) for y in range(25, 60, 5)]
        l_one = [np.array([[-3], [y], [0], [0]]) for y in range(5, 85, 5)]
        l_two = [np.array([[-8], [y], [0], [0]]) for y in range(5, 85, 5)]
        l_three = [np.array([[-13], [y], [0], [0]]) for y in range(20, 80, 5)]
        l_four = [np.array([[-18], [y], [0], [0]]) for y in range(25, 70, 5)]
        l_five = [np.array([[-23], [y], [0], [0]]) for y in range(40, 70, 5)]

        _left = [np.array([[-12], [12], [0], [0]]),
                 np.array([[-16], [20], [0], [0]]),
                 np.array([[-20], [28], [0], [0]]),
                 np.array([[-25], [30], [0], [0]]),
                 np.array([[-24], [36], [0], [0]])]

        _right = [np.array([[12], [12], [0], [0]]),
                  np.array([[16], [20], [0], [0]]),
                  np.array([[20], [28], [0], [0]]),
                  np.array([[25], [30], [0], [0]]),
                  np.array([[24], [36], [0], [0]])]

        _states = r_one + r_two + r_three + l_one + l_two + _left + _right + l_three + l_four + l_five
        _variance = 3 * np.eye(4)
        _variance[2,2] = _variance[2,2] * 2
        _variance[3,3] = _variance[3,3] * 2

        return _states, _variance
    elif model_name == 'bicycle':
        r_one = [np.array([[3], [y], [0], [0], [0]]) for y in range(5, 80, 5)]
        r_two = [np.array([[8], [y], [0], [0], [0]]) for y in range(5, 70, 5)]
        r_three = [np.array([[13], [y], [0], [0], [0]]) for y in range(25, 60, 5)]
        l_one = [np.array([[-3], [y], [0], [0], [0]]) for y in range(5, 85, 5)]
        l_two = [np.array([[-8], [y], [0], [0], [0]]) for y in range(5, 85, 5)]
        l_three = [np.array([[-13], [y], [0], [0], [0]]) for y in range(20, 80, 5)]
        l_four = [np.array([[-18], [y], [0], [0], [0]]) for y in range(25, 70, 5)]
        l_five = [np.array([[-23], [y], [0], [0], [0]]) for y in range(40, 70, 5)]

        _left = [np.array([[-12], [12], [0], [0], [0]]),
                 np.array([[-16], [20], [0], [0], [0]]),
                 np.array([[-20], [28], [0], [0], [0]]),
                 np.array([[-24], [36], [0], [0], [0]])]

        _right = [np.array([[12], [12], [0], [0], [0]]),
                  np.array([[16], [20], [0], [0], [0]]),
                  np.array([[20], [28], [0], [0], [0]]),
                  np.array([[24], [36], [0], [0], [0]])]

        _states = r_one + r_two + r_three + l_one + l_two + _left + _right + l_three + l_four + l_five
        _variance = 0.5 * np.eye(5)
        _variance[2, 2] = 10
        _variance[3, 3] = 3
        _variance[4, 4] = 3
        return _states, _variance

    elif model_name == 'kitti':
        # TODO: Make as inputs (maybe)
        grid_size = [-30, 30, -5, 45]
        nx = 30
        nz = 30
        x = np.linspace(grid_size[0], grid_size[1], nx)
        z = np.linspace(grid_size[2], grid_size[3], nz)

        t = [0, -10, 10]
        velocity_combinations = set(product(set(t), repeat=2))

        _states = []
        for i in range(len(x)):
            for j in range(len(z)):
                for vel in velocity_combinations:
                    _s = np.array([[x[i]], [z[j]], [vel[0]], [vel[1]]])
                    _states.append(_s)

        _variance = 2 * np.eye(4)
        _variance[2, 2] = _variance[2, 2] * 3
        _variance[3, 3] = _variance[2, 2] * 3

        return _states, _variance

    else:
        return None, None
