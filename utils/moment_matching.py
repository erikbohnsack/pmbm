import numpy as np


def moment_matching_dists(list_of_distributions, list_of_weights):

    normalized_weights = [float(weight)/sum(list_of_weights) for weight in list_of_weights]

    _state = np.zeros(np.shape(list_of_distributions[0].state))
    _variance = np.zeros(np.shape(list_of_distributions[0].variance))
    _weight = sum(list_of_weights)

    for index, distribution in enumerate(list_of_distributions):
        _state += distribution.state * normalized_weights[index]
    for index, distribution in enumerate(list_of_distributions):
        _delta_state = distribution.state - _state
        _variance += normalized_weights[index] * ( distribution.variance + _delta_state @ _delta_state.transpose() )
    # For numerical stability
    _variance = 0.5 * (_variance + _variance.transpose())

    return _state, _variance, _weight


def moment_matching(states_within_gate, variances_within_gate, weight_within_gate):
        normalized_weights = weight_within_gate / np.sum(weight_within_gate)
        _state = np.zeros((np.shape(states_within_gate[0])))
        _variance = np.zeros((np.shape(variances_within_gate[0])))
        _weight = sum(weight_within_gate)
        for _i in range(len(states_within_gate)):
            _state += states_within_gate[_i] * normalized_weights[_i]
        for _i in range(len(states_within_gate)):
            _delta_state = states_within_gate[_i] - _state
            _variance += normalized_weights[_i] * (variances_within_gate[_i] + _delta_state @ _delta_state.transpose())

        # For numerical stability
        _variance = 0.5 * (_variance + _variance.transpose())

        return _state, _variance, _weight
