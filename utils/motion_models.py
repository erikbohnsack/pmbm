import numpy as np
import math


class BicycleModel:
    """
    State vector:
    [   x,
        y,
        psi,
        V,
        df

    """

    def __init__(self, dt, bike_lr, bike_lf, car_lr, car_lf, pedestrian_lr, pedestrian_lf, tram_lr, tram_lf,
                 truck_lr, truck_lf, van_lr, van_lf,
                 sigma_xy_bicycle, sigma_phi, sigma_v, sigma_d):
        self.dt = dt
        self.bike_lr = bike_lr
        self.bike_lf = bike_lf
        self.car_lr = car_lr
        self.car_lf = car_lf
        self.pedestrian_lr = pedestrian_lr
        self.pedestrian_lf = pedestrian_lf
        self.tram_lr = tram_lr
        self.tram_lf = tram_lf
        self.truck_lr = truck_lr
        self.truck_lf = truck_lf
        self.van_lr = van_lr
        self.van_lf = van_lf

        self.Q = ([[sigma_xy_bicycle, 0.19450606, 0.054374531, 0.033518521, 0.0814456],
                   [0.19450606, sigma_xy_bicycle, 0.054374531, 0.033518521, 0.0814456],
                   [0.054374531, 0.054374531, sigma_phi, 0.027877464, 0.003820711],
                   [0.033518521, 0.033518521, 0.027877464, sigma_v, 0.0005],
                   [0.0814456, 0.0814456, 0.003820711, 0.0005, sigma_d]])
        # self.Q[2, 2] = sigma_phi
        # self.Q[3, 3] = sigma_v
        # self.Q[4, 4] = sigma_d
        self.model = 0

    def __call__(self, state, variance, dt=None, object_class='Car'):
        assert state.shape[0] == variance.shape[0], "State vector and covariance matrix is misaligned"
        if dt is None:
            dt = self.dt
        if object_class == 'Car':
            lr = self.car_lr
            lf = self.car_lf
        elif object_class == 'Cyclist':
            lr = self.bike_lr
            lf = self.bike_lf
        elif object_class == 'Van':
            lr = self.van_lr
            lf = self.van_lf
        elif object_class == 'Pedestrian' or object_class == 'Person':
            lr = self.pedestrian_lr
            lf = self.pedestrian_lf
        elif object_class == 'Truck':
            lr = self.truck_lr
            lf = self.truck_lf
        elif object_class == 'Tram':
            lr = self.tram_lr
            lf = self.tram_lf
        elif object_class == 'Misc':  # TODO: see what is the best lr/lf here
            lr = self.bike_lr
            lf = self.bike_lf
        else:
            raise ValueError("Input a valid object you fool. {} is certainly not one of them.".format(object_class))

        beta = math.atan2(lr * math.tan(state[4]), (lr + lf))
        x = state[0] + state[3] * math.cos(state[2] + beta) * dt
        y = state[1] + state[3] * math.sin(state[2] + beta) * dt
        psi = state[2] + state[3] / lr * math.sin(beta) * dt
        return np.array([x, y, psi, state[3], state[4]])

    def get_Q(self, object_class=None):
        return self.Q


class ConstantVelocityModel:
    """Nearly constant velocity motion model."""
    """ States = [x y v_x v_y] """

    def __init__(self, motion_noise, dt):
        """Init."""
        self.q = motion_noise
        self.dim = 4  # Dimension of this model...
        self.model = 1

        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.Q = np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0],
                           [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                           [dt ** 2 / 2, 0, dt, 0],
                           [0, dt ** 2 / 2, 0, dt]]) * self.q

    def __call__(self, state, variance, dt=None, object_class='Car'):
        """Step model."""
        # Check if the state vector is correctly represented so that matrix multiplication will work as intended...
        try:
            if not np.shape(state)[0] == self.dim:
                raise ValueError(
                    'Faulty state dimensions in CV Model! \n Need: dim{} \n Got: dim{} \n Current state: \n {}'.format(
                        self.dim, np.shape(state)[0], state))

            if not (np.shape(variance)[0] == self.dim and np.shape(variance)[1] == self.dim):
                raise ValueError(
                    'Faulty variance dimensions in CV Model! \n Need: dim{},{} \n Got: dim{},{} \n Current variance: \n {}'.format(
                        self.dim, self.dim, np.shape(variance)[0], np.shape(variance)[1], variance))
        except IndexError as e:
            print('State: {}'.format(state))
            print('Variance: {}'.format(variance))
            raise e
        if dt is None:
            F = self.F
            Q = self.Q
        else:
            F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
            Q = np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0],
                          [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                          [dt ** 2 / 2, 0, dt, 0],
                          [0, dt ** 2 / 2, 0, dt]]) * self.q
        new_state = F @ state
        new_variance = F @ variance @ F.transpose() + Q
        return new_state, new_variance

    def get_Q(self, object_class=None):
        return self.Q

    def __repr__(self):
        return '<Constant Velocity Model Class. Motion noise = {}>'.format(self.q)


class LinearWalk2D:
    def __init__(self, motion_noise):
        """Init."""
        self.dim = 2  # Dimension of this model...

        self.F = np.eye(self.dim)
        self.Q = motion_noise
        self.model = 2

    def __call__(self, state, variance, dt=None, object_class='Car'):
        # Check if the state vector is correctly represented so that matrix multiplication will work as intended...
        if not np.shape(state)[0] == self.dim:
            raise ValueError(
                'Faulty state dimensions in LW2D Model! \n Need: {} \n Got: {}'.format(self.dim, np.shape(state)[0]))

        new_state = self.F @ state
        new_variance = self.F @ variance @ self.F.transpose() + self.Q
        return new_state, new_variance


class MixedModel:
    def __init__(self, dt, motion_noise, bike_lr, bike_lf, car_lr, car_lf, pedestrian_lr, pedestrian_lf, tram_lr,
                 tram_lf,
                 truck_lr, truck_lf, van_lr, van_lf,
                 sigma_xy_bicycle, sigma_phi, sigma_v, sigma_d):
        self.BM = BicycleModel(dt=dt,
                               bike_lr=bike_lr, bike_lf=bike_lf,
                               car_lr=car_lr, car_lf=car_lf,
                               pedestrian_lr=pedestrian_lr, pedestrian_lf=pedestrian_lf,
                               tram_lr=tram_lr, tram_lf=tram_lf,
                               truck_lr=truck_lr, truck_lf=truck_lf,
                               van_lr=van_lr, van_lf=van_lf,
                               sigma_xy_bicycle=sigma_xy_bicycle,
                               sigma_phi=sigma_phi,
                               sigma_v=sigma_v,
                               sigma_d=sigma_d)

        self.CV = ConstantVelocityModel(motion_noise=motion_noise, dt=dt)
        self.model = 1

    def __call__(self, state, variance, object_class):
        if object_class == 'Pedestrian' or object_class == 'Misc' or object_class == 'Person':
            new_state, new_variance = self.CV(state=state, variance=variance, object_class=object_class)
        else:
            new_state = self.BM(state=state, variance=variance, object_class=object_class)
            new_variance = [None]

        return new_state, new_variance

    def get_Q(self, object_class):
        if object_class == 'Pedestrian' or object_class == 'Misc' or object_class == 'Person':
            return self.CV.Q
        else:
            return self.BM.Q


class ConstantAccelerationModel:
    """Nearly constant velocity motion model."""
    """ States = [x y dx dy ddx ddy] """

    def __init__(self, motion_noise, dt):
        """Init."""
        self.q = motion_noise
        self.dim = 6  # Dimension of this model...
        self.model = 1

        self.F = np.array([[1, 0, dt, 0, dt ** 2 / 2, 0],
                           [0, 1, 0, dt, 0, dt ** 2 / 2],
                           [0, 0, 1, 0, dt, 0],
                           [0, 0, 0, 1, 0, dt],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        self.Q = np.array([[dt ** 5 / 20, 0, dt ** 4 / 8, 0, dt ** 3 / 6, 0],
                           [0, dt ** 5 / 20, 0, dt ** 4 / 8, 0, dt ** 3 / 6],
                           [dt ** 4 / 8, 0, dt ** 3 / 3, 0, dt ** 2 / 2, 0],
                           [0, dt ** 4 / 8, 0, dt ** 3 / 3, 0, dt ** 2 / 2],
                           [dt ** 3 / 6, 0, dt ** 2 / 2, 0, dt, 0],
                           [0, dt ** 3 / 6, 0, dt ** 2 / 2, 0, dt]]) * self.q

    def __call__(self, state, variance, dt=None, object_class='Car'):
        """Step model."""
        # Check if the state vector is correctly represented so that matrix multiplication will work as intended...
        try:
            if not np.shape(state)[0] == self.dim:
                raise ValueError(
                    'Faulty state dimensions in CV Model! \n Need: dim{} \n Got: dim{} \n Current state: \n {}'.format(
                        self.dim, np.shape(state)[0], state))

            if not (np.shape(variance)[0] == self.dim and np.shape(variance)[1] == self.dim):
                raise ValueError(
                    'Faulty variance dimensions in CV Model! \n Need: dim{},{} \n Got: dim{},{} \n Current variance: \n {}'.format(
                        self.dim, self.dim, np.shape(variance)[0], np.shape(variance)[1], variance))
        except IndexError as e:
            print('State: {}'.format(state))
            print('Variance: {}'.format(variance))
            raise e
        if dt is None:
            F = self.F
            Q = self.Q
        else:
            F = np.array([[1, 0, dt, 0, dt ** 2 / 2, 0],
                               [0, 1, 0, dt, 0, dt ** 2 / 2],
                               [0, 0, 1, 0, dt, 0],
                               [0, 0, 0, 1, 0, dt],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]])

            Q = np.array([[dt ** 5 / 20, 0, dt ** 4 / 8, 0, dt ** 3 / 6, 0],
                               [0, dt ** 5 / 20, 0, dt ** 4 / 8, 0, dt ** 3 / 6],
                               [dt ** 4 / 8, 0, dt ** 3 / 3, 0, dt ** 2 / 2, 0],
                               [0, dt ** 4 / 8, 0, dt ** 3 / 3, 0, dt ** 2 / 2],
                               [dt ** 3 / 6, 0, dt ** 2 / 2, 0, dt, 0],
                               [0, dt ** 3 / 6, 0, dt ** 2 / 2, 0, dt]]) * self.q
        new_state = F @ state
        new_variance = F @ variance @ F.transpose() + Q
        return new_state, new_variance

    def get_Q(self, object_class=None):
        return self.Q

    def __repr__(self):
        return '<Constant Acceleration Model Class. Motion noise = {}>'.format(self.q)