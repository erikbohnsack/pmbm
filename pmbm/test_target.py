import unittest
import numpy as np
from .target import Target

target_id = 0
time_of_birth = 0
state = np.array([[-2], [-2]])
variance = np.eye(2)
measurement_index = 0
motion_model = np.eye(2)
survival_probability = 0.5
detection_probability = 0.5
measurement_model = np.eye(2)
measurement_noise = np.eye(2)
gating_distance = 2


class TestTarget(unittest.TestCase):
    def test_hypo_addition(self):
        target = Target(target_id, time_of_birth, state, variance, measurement_index, motion_model,
                        survival_probability, detection_probability, measurement_model,
                        measurement_noise, gating_distance)
        assert(len(target.single_target_hypotheses) == 1)

    def test_len_new_hypos(self):
        target = Target(target_id, time_of_birth, state, variance, measurement_index, motion_model,
                        survival_probability, detection_probability, measurement_model,
                        measurement_noise, gating_distance)
        measurements = [np.array([[-2], [-2]]), np.array([[100], [100]])]
        assert len(target.single_target_hypotheses) == 1
        target.new_hypos(measurements, 1)
        assert len(target.single_target_hypotheses) == 2
        target.new_hypos(measurements, 2)
        assert len(target.single_target_hypotheses) == 4
