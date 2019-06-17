import unittest
import numpy as np
from .poisson import Distribution, Poisson
from utils import motion_models


class TestDistribution(unittest.TestCase):
    def test_predict_2D(self):
        init_state = np.random.rand(2, 1)
        init_variance = np.random.rand(2, 2)
        init_weight = 1
        distribution = Distribution(init_state, init_variance, init_weight)
        Q = np.matrix('1, 0 ; 0, 1')
        motion_model = motion_models.LinearWalk2D(motion_factor=1, motion_noise=Q)
        survival_probability = 0.9
        distribution.predict(motion_model, survival_probability)
        self.assertTrue((init_state == distribution.state).all())
        self.assertTrue((init_variance + Q == distribution.variance).all())


class TestPoisson(unittest.TestCase):
    def test_give_birth(self):
        poisson = Poisson(birth_var=np.eye(2), birth_state=[np.array([[-2], [-2]]), np.array([[2], [2]]), np.array([[-4], [-4]]),
                          np.array([[4], [4]])],)
        self.assertTrue(len(poisson.distributions) == 0)
        poisson.give_birth()
        self.assertTrue(len(poisson.distributions) == 4)

    def test_predict(self):
        poisson = Poisson(birth_var=np.eye(2), birth_state=[np.array([[-2], [-2]]), np.array([[2], [2]]), np.array([[-4], [-4]]),
                          np.array([[4], [4]])])
        Q = np.matrix('1, 0 ; 0, 1')
        motion_model = motion_models.LinearWalk2D(motion_factor=1, motion_noise=Q)

        survival_probability = 0.8
        poisson.predict(motion_model, survival_probability)
        self.assertTrue(len(poisson.distributions) == 4)
        poisson.predict(motion_model, survival_probability)
        self.assertTrue(len(poisson.distributions) == 8)
        self.assertTrue((poisson.distributions[1].state == motion_model(poisson.distributions[5].state, Q)[0]).all())
        self.assertTrue((poisson.distributions[0].state == motion_model(poisson.distributions[4].state, Q)[0]).all())

    def test_merge(self):
        state1 = np.array([[-2.], [-2.]])
        state2 = np.array([[2.], [2.]])
        state3 = np.array([[1.], [2.]])
        cov = np.eye(2)
        distr1 = Distribution(state1, cov, 1.)
        distr2 = Distribution(state2, cov, 2.)
        distr3 = Distribution(state3, cov, 3.)
        poisson = Poisson(birth_state=np.random.rand(2, 1),
                          birth_var=np.eye(2),
                          birth_weight_factor=1,
                          prune_threshold=0.1,
                          merge_threshold=2,
                          reduce_factor=0.1)
        poisson.distributions = [distr1, distr2, distr3]

        assert len(poisson.distributions) == 3
        poisson.merge()
        assert len(poisson.distributions) == 2
        poisson.merge()
        assert len(poisson.distributions) == 2
        poisson.merge_threshold = 10
        poisson.merge()
        assert len(poisson.distributions) == 1


if __name__ == '__main__':
    unittest.main()
