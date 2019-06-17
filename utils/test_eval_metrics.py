import unittest
from utils.eval_metrics import GOSPA
from data_utils.generate_data import GenerateData
import numpy as np


class TestEvalMetrics(unittest.TestCase):
    def test_gospa(self):
        true_trajectories, measurements = GenerateData.generate_2D_data(
            mutation_probability=0.4,
            measurement_noise=0.4,
            missed_detection_probability=0.05,
            clutter_probability=0.1)
        #GenerateData.plot_generated_data(true_trajectories, measurements)
        for t in range(len(measurements)):
            # Vill vstacka
            ground_truths = np.array([], dtype=np.float).reshape(0, 2)
            estimates = np.array([], dtype=np.float).reshape(0, 2)
            for i in range(len(true_trajectories)):
                ground_truths = np.vstack((ground_truths, true_trajectories[i][t][0:2].reshape(1, 2)))
            for m in measurements[t]:
                if m.size:
                    estimates = np.vstack((estimates, m.reshape(1, 2)))


            score = GOSPA(ground_truths, estimates, p=1, c=100, alpha=2.)
            print(score)

