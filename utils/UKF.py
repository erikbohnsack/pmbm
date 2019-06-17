import numpy as np
from filterpy.kalman import unscented_transform
from scipy.linalg import cholesky
from utils.matrix_stuff import switch_state_direction


class UnscentedKalmanFilter:
    def __init__(self, dt, dim_x, dim_z, points, motion_model, measurement_model,
                 residual_z=None, residual_x=None, sqrt_fn=None):

        self.points_fn = points
        self._num_sigmas = points.num_sigmas()
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.measurement_model = measurement_model
        self.motion_model = motion_model
        self._dt = dt
        self._num_sigmas = points.num_sigmas()

        # weights for the means and covariances.
        self.Wm, self.Wc = points.Wm, points.Wc

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

    def __repr__(self):
        pass

    def predict(self, state, variance, motion_model, motion_noise, object_class, dt=None, UT=None):
        """
        Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '
        Important: this MUST be called before update() is called for the first
        time.
        Parameters
        ----------
        state  : Prior state estimate (x_{k-1|k-1})
        variance  : Prior state covariance matrix (P_{k-1|k-1})
        dt : double, optional
            If specified, the time step to be used for this prediction.
            self._dt is used if this is not provided.
        motion_model : callable f(x, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.
        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.
        object_class : Either Car or Bicycle.
        """
        assert state.shape == (self._dim_x, ), "State shape not 1D array, required by UKF"
        assert state.shape[0] == variance.shape[0], "State vector and covariance matrix is misaligned"
        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        points = self.compute_process_sigmas(x=state, P=variance, dt=dt, motion_model=motion_model, object_class=object_class)

        # and pass sigmas through the unscented transform to compute prior
        x, P = UT(points, self.Wm, self.Wc, motion_noise, mean_fn=None, residual_fn=self.residual_x)
        return x.reshape(-1, 1), P

    def update(self, state, variance, measurement, measurement_model, measurement_noise, UT=None):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.
        Parameters
        ----------
        state  : Predicted state estimate (x_{k|k-1})
        variance  : Predicted state covariance matrix (P_{k|k-1})
        measurement : numpy.array of shape (dim_z)
            measurement vector
        measurement_noise : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.
        measurement_model:
        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        """
        assert state.shape == (self._dim_x,), "State shape not 1D array, required by UKF"
        assert state.shape[0] == variance.shape[0], "State vector and covariance matrix is misaligned"
        if measurement is None:
            return state, variance

        if measurement_model is None:
            measurement_model = self.measurement_model

        if UT is None:
            UT = unscented_transform

        if measurement_noise is None:
            measurement_noise = self.R
        elif np.isscalar(measurement_noise):
            measurement_noise = np.eye(self._dim_z) * measurement_noise

        sigmas_f = self.points_fn.sigma_points(state, variance)

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in sigmas_f:
            sigmas_h.append(measurement_model @ s)

        sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, S = UT(sigmas_h, self.Wm, self.Wc, measurement_noise, mean_fn=None, residual_fn=self.residual_z)
        SI = np.linalg.inv(S)
        Pxz = self.cross_variance(state, zp, sigmas_f, sigmas_h)

        K = Pxz @  SI  # Kalman gain
        y = self.residual_z(measurement[:, 0], zp)  # residual

        state += K @ y
        #if state[3] < 0:
        #    print("------------ ERROR -------------")

        #    print("State: {}".format(state))
        #    print("Old State: {}".format(state - K @ y))
        #    print("Measurement: {}".format(measurement))
        #    print("K: {}".format(K))
        #    print("Pxz: {}".format(Pxz))
        #    print("SI: {}".format(SI))
        #    print("y: {}".format(y))
        #   raise ValueError("Negative speed")

        variance -= K @ S @ K.T
        #output_state = switch_state_direction(state)
        return state.reshape(-1, 1), variance

    def compute_process_sigmas(self, x, P, dt, motion_model=None, object_class='Car'):
            """
            computes the values of sigmas_f. Normally a user would not call
            this, but it is useful if you need to call update more than once
            between calls to predict (to update for multiple simultaneous
            measurements), so the sigmas correctly reflect the updated state
            x, P.
            """
            if motion_model is None:
                motion_model = self.motion_model
            # calculate sigma points for given mean and covariance
            sigmas = self.points_fn.sigma_points(x, P)
            sigmas_f = np.zeros((self._num_sigmas, self._dim_x))
            for i, s in enumerate(sigmas):
                if motion_model.model:
                    sigmas_f[i] = motion_model(state=s, variance=P, object_class=object_class)[0]
                else:
                    sigmas_f[i] = motion_model(state=s, variance=P, object_class=object_class)
            return sigmas_f

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `z`.
        """

        Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dz = self.residual_z(sigmas_h[i], z)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        return Pxz

