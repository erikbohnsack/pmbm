import numpy as np
from utils.UKF import UnscentedKalmanFilter
from scipy.stats import multivariate_normal


class LinearFilter:
    def __init__(self):
        pass

    def predict(self, state, variance, motion_model, motion_noise=None, object_class=None):
        assert state.shape[1] == 1, "State is not a column vector"
        assert state.shape[0] == variance.shape[0], "State vector and covariance matrix is misaligned"

        _state, _variance = motion_model(state=state, variance=variance, object_class=object_class)

        # For numerical stability
        _variance = 0.5 * (_variance + _variance.transpose())
        return _state, _variance

    def update(self, state, variance, measurement, measurement_model, measurement_noise, object_class=None):
        assert state.shape[1] == 1, "State is not a column vector"
        assert state.shape[0] == variance.shape[0], "State vector and covariance matrix is misaligned"

        # Else we update all the single target hypotheses
        # S = H * P * H' + R
        _S = measurement_model @ variance @ measurement_model.transpose() + measurement_noise
        # For numerical stability
        _S = 0.5 * (_S + _S.transpose())
        # W = P * H'
        _W = variance @ measurement_model.transpose()
        # K = W * inv(S)
        _K = _W @ np.linalg.inv(_S)
        # P = P - K
        _P = variance - _K @ _W.transpose()
        # For numerical stability
        _P = 0.5 * (_P + _P.transpose())

        _v = measurement - measurement_model @ state

        _state = state + _K @ _v
        _variance = _P

        reshaped_meas = measurement.reshape(np.shape(measurement_model)[0], )
        reshaped_mean = (measurement_model @ _state).reshape(np.shape(measurement_model)[0], )

        meas_likelihood = multivariate_normal.pdf(reshaped_meas, reshaped_mean, _S)
        return _state, _variance, meas_likelihood


class UKF(UnscentedKalmanFilter):
    def __init__(self, dt, dim_x, dim_z, points, fx, hx, residual_z=None, residual_x=None, sqrt_fn=None):
        super().__init__(dt, dim_x, dim_z, points, fx, hx, residual_z=residual_z,
                         residual_x=residual_x, sqrt_fn=sqrt_fn)

    def predict(self, state, variance, motion_model, motion_noise, object_class, dt=None, UT=None):
        assert state.shape[0] == self._dim_x, "Mismatch in state cardinality"
        assert state.shape[1] == 1, "State is not a column vector"
        assert state.shape[0] == variance.shape[0], "State vector and covariance matrix is misaligned"
        _state, _variance = super().predict(state.reshape(len(state),), variance, motion_model, motion_noise,
                                            object_class, dt=dt, UT=UT)
        _variance = 0.5 * (_variance + _variance.transpose())
        return _state, _variance

    def update(self, state, variance, measurement, measurement_model, measurement_noise, oject_class=None, UT=None):
        assert state.shape[0] == self._dim_x, "Mismatch in state cardinality"
        assert state.shape[1] == 1, "State is not a column vector"
        assert state.shape[0] == variance.shape[0], "State vector and covariance matrix is misaligned"
        _state, _variance = super().update(state.reshape(len(state),), variance, measurement, measurement_model,
                                           measurement_noise, UT=UT)
        reshaped_meas = measurement.reshape(np.shape(measurement_model)[0], )
        reshaped_mean = (measurement_model @ state).reshape(np.shape(measurement_model)[0], )
        _S = measurement_model @ variance @ measurement_model.transpose() + measurement_noise  # Input variance here
        _S = 0.5 * (_S + _S.transpose())
        meas_likelihood = multivariate_normal.pdf(reshaped_meas, reshaped_mean, _S)
        _variance = 0.5 * (_variance + _variance.transpose())
        return _state, _variance, meas_likelihood


class MixedFilter():
    def __init__(self, dt, dim_x, dim_z, points, fx, hx, residual_z=None, residual_x=None, sqrt_fn=None):
        self.LF = LinearFilter()
        self.UKF = UKF(dt=dt, dim_x=dim_x, dim_z=dim_z, points=points, fx=fx, hx=hx,
                       residual_z=residual_z, residual_x=residual_x, sqrt_fn=sqrt_fn)

    def predict(self, state, variance, motion_model, motion_noise, object_class, dt=None, UT=None):
        if object_class == 'Pedestrian' or object_class == 'Misc' or object_class == 'Person':
            _state, _variance = self.LF.predict(state=state, variance=variance, motion_model=motion_model,
                                                object_class=object_class)
        else:
            _state, _variance = self.UKF.predict(state=state, variance=variance, motion_model=motion_model,
                                                 motion_noise=motion_noise, object_class=object_class, dt=dt, UT=UT)
        return _state, _variance

    def update(self, state, variance, measurement, measurement_model, measurement_noise, object_class, UT=None):
        if object_class == 'Pedestrian' or object_class == 'Misc' or object_class == 'Person':
            _state, _variance, meas_likelihood = self.LF.update(state=state, variance=variance,
                                                                measurement=measurement,
                                                                measurement_model=measurement_model,
                                                                measurement_noise=measurement_noise)
        else:
            _state, _variance, meas_likelihood = self.UKF.update(state=state, variance=variance,
                                                                 measurement=measurement,
                                                                 measurement_model=measurement_model,
                                                                 measurement_noise=measurement_noise, UT=UT)
        return _state, _variance, meas_likelihood

