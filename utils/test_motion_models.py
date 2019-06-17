import numpy as np
import unittest
from .motion_models import BicycleModel


class TestBicycle(unittest.TestCase):
    def test_propagate(self):
        dt = 1
        bike_lr, bike_lf, car_lr, car_lf, sigma_phi, sigma_v, sigma_d = (1., 1., 1., 1., 1., 1., 1.)
        model = BicycleModel(dt, bike_lr, bike_lf, car_lr, car_lf, sigma_phi, sigma_v, sigma_d)
        state = np.array([1., 1., 0, 1., 0]).reshape(5, 1)
        variance = np.eye(5)
        _state = model(state, variance)
        assert state[0, 0] + 1 == _state[0, 0]

        sate = np.array([1., 1., 0, -1., 0]).reshape(5, 1)
        _sate = model(sate, variance)
        assert state[0, 0] - 1 == _sate[0, 0]

