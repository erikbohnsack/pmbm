import numpy as np
import unittest
from .mot_metrics import MotCalculator
from pmbm.single_target_hypothesis import SingleTargetHypothesis


class TestMotMetrics(unittest.TestCase):
    def testie(self):
        acc = MotCalculator(5)
        target_id = 0
        single_target_id = 0
        test_state = np.array([[0], [35]])
        test_var = np.eye(2)
        single_target = SingleTargetHypothesis(measurement_index=1,
                                               state=test_state,
                                               variance=test_var,
                                               existence=1.,
                                               weight=1.,
                                               time_of_birth=1)
        estimated_target = {}
        estimated_target['target_idx'] = target_id
        estimated_target['single_hypo_idx'] = single_target_id
        estimated_target['single_target'] = single_target
        estimated_targets = [estimated_target]
        acc.calculate(estimated_targets, 3)
        acc.calculate(estimated_targets, 4)

        #print(acc.events)
        #mh = mm.metrics.create()

        #summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')