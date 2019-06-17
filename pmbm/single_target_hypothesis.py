from utils.constants import LARGE


class SingleTargetHypothesis:
    def __init__(self, measurement_index, state, variance, existence, weight, time_of_birth, single_cost=LARGE):
        assert state.shape[1] == 1, "Input state is not column vector"
        assert state.shape[0] == variance.shape[0], "State vector not aligned with Covariance matrix"
        assert isinstance(weight, float), "Input weight not a float. Current type: {}".format(type(weight))

        self.state = state
        self.variance = variance
        self.existence = existence
        self.weight = weight
        self.single_cost = single_cost
        self.measurement_index = measurement_index
        self.children = []
        self.time_of_birth = time_of_birth

    def __repr__(self):
        if self.state.shape[0] == 5:
            return '\t<SingleTargetHypothesis | t_birth: {}, \tw.: {}, \tc.: {}, \tP_ex.: {}, \tmeas_idx: {}, \tX: {}>\n'.format(
                self.time_of_birth,
                round(self.weight, 2),
                round(self.single_cost, 2),
                round(self.existence, 2),
                self.measurement_index,
                (round(float(self.state[0]), 2), round(float(self.state[1]), 2), round(float(self.state[2]), 2),
                 round(float(self.state[3]), 2), round(float(self.state[4]), 2)))
        else:
            return '\t<SingleTargetHypothesis | t_birth: {}, \tw.: {}, \tc.: {}, \tP_ex.: {}, \tmeas_idx: {}, \tX: {}>\n'.format(
                self.time_of_birth,
                round(self.weight, 2),
                round(self.single_cost, 2),
                round(self.existence, 2),
                self.measurement_index,
                (round(float(self.state[0]), 2), round(float(self.state[1]), 2), round(float(self.state[2]), 2),
                 round(float(self.state[3]), 2)))
