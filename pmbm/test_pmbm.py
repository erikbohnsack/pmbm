import unittest
import numpy as np
from .pmbm import PMBM, Settings
from .target import Target
from utils.constants import LARGE
from .global_hypothesis import GlobalHypothesis

# Test Target

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


class TestPMBM(unittest.TestCase):
    def test_possible_new_targets(self):
        pass  # possible_new_targets(self, measurements):

    def test_cost_new_targets(self):
        max_nof_global_hypos = 2

        measurement_model = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        measurement_noise = 0.1 * np.eye(2)
        settings = Settings(detection_probability=0.95,
                            survival_probability=0.95,
                            prune_threshold_poisson=0,
                            prune_threshold_targets=0,
                            gating_distance=15,
                            birth_gating_distance=15,
                            max_nof_global_hypos=max_nof_global_hypos,
                            motion_model='CV',
                            motion_noise=1,
                            measurement_model=measurement_model,
                            measurement_noise=measurement_noise)
        pmbm = PMBM(settings)
        measurements = [np.array([[-45], [-45]]), np.array([[-45], [45]])]
        nof_measurements = len(measurements)

        #  NO NEW TARGETS.
        W_nt, new_target_map, new_measurement_map_meas2row, new_measurement_map_row2meas = pmbm.new_targets_cost(
            measurements)
        assert (W_nt == np.full((nof_measurements, nof_measurements), LARGE)).all()
        assert new_target_map == {}
        assert new_measurement_map_row2meas == {}
        assert new_measurement_map_meas2row == {}

        # NEW TARGET
        pmbm.current_time = 10

        jx = 1
        _state = np.array([[-45], [45]])
        _variance = np.eye(2)
        _weight = 10.
        _existence = 1

        target_id = 100
        target = Target(measurement_index=jx, time_of_birth=pmbm.current_time,
                        state=_state,
                        variance=_variance,
                        weight=_weight,
                        existence=_existence,
                        motion_model=pmbm.motion_model,
                        survival_probability=pmbm.survival_probability,
                        detection_probability=pmbm.detection_probability,
                        measurement_model=pmbm.measurement_model,
                        measurement_noise=pmbm.measurement_noise,
                        gating_distance=pmbm.gating_distance)

        pmbm.targets[target_id] = target
        pmbm.new_targets = [target_id]
        W_nt, new_target_map, new_measurement_map_meas2row, new_measurement_map_row2meas = pmbm.new_targets_cost(
            measurements)
        assert W_nt[0, 0] == - _weight
        assert new_target_map[target_id] == 0
        assert new_measurement_map_meas2row[jx] == 0
        assert new_measurement_map_row2meas[0] == jx

    def test_cap_global_hypos(self):
        max_nof_global_hypos = 2

        measurement_model = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        measurement_noise = 0.1 * np.eye(2)

        pmbm = PMBM(state_dims=4,
                    detection_probability=0.95,
                    survival_probability=0.95,
                    prune_threshold_poisson=0,
                    prune_threshold_targets=0,
                    gating_distance=15,
                    birth_gating_distance=15,
                    max_nof_global_hypos=max_nof_global_hypos,
                    motion_model='CV',
                    motion_noise=1,
                    measurement_model=measurement_model,
                    measurement_noise=measurement_noise)

        pmbm.global_hypotheses[1] = GlobalHypothesis(1.0, [])
        pmbm.global_hypotheses[2] = GlobalHypothesis(0.8, [])
        pmbm.global_hypotheses[3] = GlobalHypothesis(0.5, [])
        pmbm.global_hypotheses[4] = GlobalHypothesis(1.2, [])

        pmbm.cap_global_hypos()

        assert len(pmbm.global_hypotheses) == max_nof_global_hypos, "Capping failed"
        assert pmbm.global_hypotheses[1], "Global hypo with key = 1 shouldn't have been removed"
        assert pmbm.global_hypotheses[4], "Global hypo with key = 4 shouldn't have been removed"

    def test_prune_global_hypo(self):
        pmbm = PMBM(prune_threshold_global_hypo=2, motion_model='CV')
        pmbm.global_hypotheses = {0: GlobalHypothesis(weight=1, hypothesis=None),
                                  1: GlobalHypothesis(weight=3, hypothesis=None)}
        assert len(pmbm.global_hypotheses) == 2
        pmbm.prune_global()
        assert len(pmbm.global_hypotheses) == 1

    def test_prune_targets(self):
        pmbm = PMBM(prune_threshold_targets=3, motion_model='CV')
        pmbm.targets = {0: Target(time_of_birth=0,
                                  state=np.array([[1], [1]]),
                                  variance=np.eye(2),
                                  weight=1.,
                                  existence=1,
                                  measurement_index=1,
                                  motion_model=1,
                                  survival_probability=1,
                                  detection_probability=1,
                                  measurement_model=1,
                                  measurement_noise=1,
                                  gating_distance=1,
                                  verbose=False),
                        1: Target(time_of_birth,
                                  state=np.array([[1], [1]]),
                                  variance=np.eye(2),
                                  weight=1.,
                                  existence=0.5,
                                  measurement_index=1,
                                  motion_model=1,
                                  survival_probability=1,
                                  detection_probability=1,
                                  measurement_model=1,
                                  measurement_noise=1,
                                  gating_distance=1,
                                  verbose=False)}
        pmbm.global_hypotheses = {0: GlobalHypothesis(weight=1, hypothesis=[(0, 0)]),
                                  1: GlobalHypothesis(weight=10, hypothesis=[(1, 0)]),
                                  2: GlobalHypothesis(weight=1, hypothesis=[(0, 0), (1, 0)])}

        assert len(pmbm.targets) == 2
        pmbm.recycle_targets()
        assert len(pmbm.targets) == 1

    def test_recycling(self):
        init_state = np.array([[1], [1]])
        init_cov = np.eye(2)
        pmbm = PMBM(prune_threshold_targets=3, motion_model='CV')
        pmbm.targets = {0: Target(time_of_birth=0,
                                  state=init_state,
                                  variance=init_cov,
                                  weight=1.,
                                  existence=1,
                                  measurement_index=1,
                                  motion_model=1,
                                  survival_probability=1,
                                  detection_probability=1,
                                  measurement_model=1,
                                  measurement_noise=1,
                                  gating_distance=1,
                                  verbose=False),
                        1: Target(time_of_birth,
                                  state=init_state,
                                  variance=init_cov,
                                  weight=1.,
                                  existence=0.5,
                                  measurement_index=1,
                                  motion_model=1,
                                  survival_probability=1,
                                  detection_probability=1,
                                  measurement_model=1,
                                  measurement_noise=1,
                                  gating_distance=1,
                                  verbose=False)}
        pmbm.global_hypotheses = {0: GlobalHypothesis(weight=1, hypothesis=[(0, 0)]),
                                  1: GlobalHypothesis(weight=10, hypothesis=[(1, 0)]),
                                  2: GlobalHypothesis(weight=1, hypothesis=[(0, 0), (1, 0)])}

        assert len(pmbm.poisson.distributions) == 0
        pmbm.recycle_targets()
        assert len(pmbm.poisson.distributions) == 1
        assert (pmbm.poisson.distributions[0].state == init_state).all()
        assert (pmbm.poisson.distributions[0].variance == init_cov).all()


if __name__ == '__main__':
    unittest.main()
