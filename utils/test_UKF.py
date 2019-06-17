import numpy as np
import unittest
from .UKF import UnscentedKalmanFilter as UKF
from .motion_models import BicycleModel
from filterpy.kalman import MerweScaledSigmaPoints
from math import pi


class TestUKF(unittest.TestCase):
    def test_propagate(self):
        dt = 1
        bike_lr, bike_lf, car_lr, car_lf, sigma_phi, sigma_v, sigma_d = (1., 1., 1., 1., 1., 1., 1.)
        model = BicycleModel(dt, bike_lr, bike_lf, car_lr, car_lf, sigma_phi, sigma_v, sigma_d)


        dim_x = 5
        dim_z = 2
        points = MerweScaledSigmaPoints(dim_x, alpha=1, beta=2., kappa=2)
        self.assertAlmostEqual(sum(points.Wm), 1)
        print()
        print(points.Wm)
        dt = 1
        measurement_model = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        measurement_noise = 1 * np.eye(2)
        state = np.array([1., 1., pi/2, 1., 0])
        variance = np.eye(dim_x) * 0.2  # initial uncertainty\
        kf = UKF(dim_x=dim_x, dim_z=dim_z, dt=dt, points=points, motion_model=model, measurement_model=measurement_model)

        z_std = 0.1
        kf.R = np.diag([z_std ** 2, z_std ** 2])  # 1 standard
        kf.Q = np.eye(dim_x)
        zs = [[i + np.random.randn() * z_std, i + np.random.randn() * z_std] for i in range(50)]  # measurements

        _state, _var = kf.predict(state, variance, model, motion_noise=kf.Q, object_class='Car')
        print("State: {}".format(_state))
        print("Var: {}".format(_var))
        #hoppsan = kf.update(x, P, measurement)


