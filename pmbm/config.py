import numpy as np
from utils import motion_models
from utils import poisson_birth_state_models
from utils.coord_transf import coordinate_transform_bicycle, coordinate_transform_CV, coordinate_transform_mixed, coordinate_transform_CA
from .filter import LinearFilter, UKF, MixedFilter
from filterpy.kalman import MerweScaledSigmaPoints


class Config:
    def __init__(self,
                 config_name='NA',
                 detection_probability=0.9,
                 survival_probability=0.9,
                 prune_threshold_poisson=0.2,
                 prune_threshold_global_hypo=-6,
                 prune_threshold_targets=-6,
                 prune_single_existence=1e-4,
                 clutter_intensity=1e-4,
                 poisson_merge_threshold=2,
                 poisson_reduce_factor=0.1,
                 uniform_weight=1,
                 uniform_radius=100,
                 uniform_angle=(0.78, 2.35),
                 uniform_adjust=5.,
                 gating_distance=3,
                 measurement_var_xy=1e-1,
                 measurement_var_psi=0.5,
                 birth_gating_distance=9,
                 birth_weight=5e-3,
                 global_init_weight=1,
                 desired_nof_global_hypos=20,
                 max_nof_global_hypos=25,
                 min_new_nof_global_hypos=1,
                 max_new_nof_global_hypos=10,
                 motion_model='Bicycle',
                 poisson_states_model_name='uniform',
                 filter_class='UKF',
                 bike_lr=0.5,
                 bike_lf=0.75,
                 car_lr=2,
                 car_lf=2,
                 pedestrian_lr=0.5,
                 pedestrian_lf=0.5,
                 tram_lr=10,
                 tram_lf=2,
                 truck_lr=10,
                 truck_lf=2,
                 van_lr=3,
                 van_lf=2,
                 dt=0.1,
                 ukf_alpha=0.01,
                 ukf_beta=4.,
                 ukf_kappa=1e5,
                 sigma_cv=2,
                 sigma_ca=2,
                 sigma_xy_bicycle=0.5,
                 sigma_v=1,
                 sigma_d=0.01,
                 sigma_phi=1,
                 poisson_v=10,
                 poisson_d=5,
                 poisson_vx=10,
                 poisson_vy=10,
                 poisson_mean_v=0,
                 poisson_mean_d=0,
                 poisson_mean_vx=0,
                 poisson_mean_vy=0,
                 show_predictions=10,
                 coord_transform=True,
                 classes_to_track=['all']):

        self.name = config_name

        if motion_model == 'CV' or motion_model == 'CV-Kitti':
            self.measurement_model = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            self.measurement_noise = measurement_var_xy * np.eye(2)
        elif motion_model == 'Bicycle':
            self.measurement_model = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -1, 0, 0]])
            self.measurement_noise = np.eye(3)
            self.measurement_noise[0, 0] = measurement_var_xy
            self.measurement_noise[1, 1] = measurement_var_xy
            self.measurement_noise[2, 2] = measurement_var_psi
        elif motion_model == 'CA':
            self.measurement_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
            self.measurement_noise = measurement_var_xy * np.eye(2)
        elif motion_model == 'Mixed':
            temp_measurement_noise = np.eye(3)
            temp_measurement_noise[0, 0] = measurement_var_xy
            temp_measurement_noise[1, 1] = measurement_var_xy
            temp_measurement_noise[2, 2] = measurement_var_psi
            self.measurement_models = {'CV': np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
                                       'Bicycle': np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -1, 0, 0]])}
            self.measurement_noises = {'CV': measurement_var_xy * np.eye(2),
                                       'Bicycle': temp_measurement_noise}
            self.measurement_model = None
            self.measurement_noise = None
        else:
            raise ValueError(
                'No measurement model! Measurement model for Motion Model: {} not added '.format(motion_model))

        self.detection_probability = detection_probability
        self.survival_probability = survival_probability
        self.prune_threshold_poisson = prune_threshold_poisson
        self.prune_threshold_global_hypo = prune_threshold_global_hypo
        self.prune_threshold_targets = prune_threshold_targets
        self.prune_single_existence = prune_single_existence
        self.gating_distance = gating_distance
        self.birth_gating_distance = birth_gating_distance
        self.birth_weight = birth_weight
        self.desired_nof_global_hypos = desired_nof_global_hypos
        self.max_nof_global_hypos = max_nof_global_hypos
        self.min_new_nof_global_hypos = min_new_nof_global_hypos
        self.max_new_nof_global_hypos = max_new_nof_global_hypos
        self.global_init_weight = global_init_weight
        self.clutter_intensity = clutter_intensity
        self.poisson_merge_threshold = poisson_merge_threshold
        self.poisson_reduce_factor = poisson_reduce_factor
        self.show_predictions = show_predictions
        self.dt = dt
        self.classes_to_track = classes_to_track
        self.motion_model_name = motion_model
        self.poisson_states_model_name = poisson_states_model_name
        self.filter_name = filter_class
        if motion_model == 'CV':
            self.state_dims = 4
            self.motion_model = motion_models.ConstantVelocityModel(motion_noise=0.5, dt=dt)
            self.motion_noise = self.motion_model.Q
            self.coord_transform = None

        elif motion_model == 'CV-Kitti':
            self.state_dims = 4
            self.motion_model = motion_models.ConstantVelocityModel(motion_noise=sigma_cv, dt=dt)
            self.motion_noise = self.motion_model.Q
            if coord_transform:
                self.coord_transform = coordinate_transform_CV
            else:
                self.coord_transform = None

        # CONSTANT ACCERLERATION
        elif motion_model == 'CA':
            self.state_dims = 6
            self.motion_model = motion_models.ConstantAccelerationModel(motion_noise=sigma_ca, dt=dt)
            self.motion_noise = self.motion_model.Q
            if coord_transform:
                self.coord_transform = coordinate_transform_CA
            else:
                self.coord_transform = None


        elif motion_model == 'LW2D':
            self.state_dims = 2
            self.motion_model = motion_models.LinearWalk2D(motion_noise=0.5)
            self.motion_noise = self.motion_model.Q
            self.coord_transform = None

        elif motion_model == 'Bicycle':
            self.state_dims = 5
            self.motion_model = motion_models.BicycleModel(dt=dt,
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
            self.motion_noise = self.motion_model.Q
            if coord_transform:
                self.coord_transform = coordinate_transform_bicycle
            else:
                self.coord_transform = None

        elif motion_model == 'Mixed':
            self.state_dims = 5
            self.motion_model = motion_models.MixedModel(dt=dt, motion_noise=2,
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
            if coord_transform:
                self.coord_transform = coordinate_transform_mixed
            else:
                self.coord_transform = None


        else:
            raise ValueError('Choose a motion model that exists you fool!')

        if poisson_states_model_name == 'uniform' or \
                poisson_states_model_name == 'uniform-CV' or \
                poisson_states_model_name == 'uniform-CA' or \
                poisson_states_model_name == 'uniform-mixed':
            self.uniform_radius = uniform_radius
            self.uniform_weight = uniform_weight
            self.uniform_angle = uniform_angle
            self.uniform_adjust = uniform_adjust
            if motion_model == 'Mixed':
                uniform_covariance_cv = np.zeros((np.shape(self.measurement_models['CV'])[1],
                                                  np.shape(self.measurement_models['CV'])[1]))
                uniform_covariance_bicycle = np.zeros((np.shape(self.measurement_models['Bicycle'])[1],
                                                       np.shape(self.measurement_models['Bicycle'])[1]))

                uniform_covariance_cv[0:np.shape(self.measurement_noises['CV'])[0],
                0:np.shape(self.measurement_noises['CV'])[0]] = self.measurement_noises['CV']
                uniform_covariance_cv[-2, -2] = poisson_vx
                uniform_covariance_cv[-1, -1] = poisson_vy

                uniform_covariance_bicycle[0:np.shape(self.measurement_noises['Bicycle'])[0],
                0:np.shape(self.measurement_noises['Bicycle'])[0]] = self.measurement_noises['Bicycle']
                uniform_covariance_bicycle[-2, -2] = poisson_v
                uniform_covariance_bicycle[-1, -1] = poisson_d

                self.uniform_covariances = {'CV': uniform_covariance_cv,
                                            'Bicycle': uniform_covariance_bicycle}
                self.unmeasurable_state_means = {'CV': np.array([[poisson_mean_vx], [poisson_mean_vy]]),
                                                 'Bicycle': np.array([[poisson_mean_v], [poisson_mean_d]])}

            else:
                self.uniform_covariance = np.zeros((self.state_dims, self.state_dims))
                self.uniform_covariance[0:self.measurement_noise.shape[0],
                0:self.measurement_noise.shape[1]] = self.measurement_noise
                if motion_model == 'CV' or motion_model == 'CV-Kitti':
                    self.uniform_covariance[-2, -2] = poisson_vx
                    self.uniform_covariance[-1, -1] = poisson_vy
                    self.unmeasurable_state_mean = np.array([[poisson_mean_vx], [poisson_mean_vy]])
                elif motion_model == 'CA':
                    self.uniform_covariance[-4, -4] = poisson_vx
                    self.uniform_covariance[-3, -3] = poisson_vy
                    self.uniform_covariance[-2, -2] = poisson_vx
                    self.uniform_covariance[-1, -1] = poisson_vy
                    self.unmeasurable_state_mean = np.array([[poisson_mean_vx], [poisson_mean_vy],
                                                             [poisson_mean_vx], [poisson_mean_vy]])
                else:
                    self.uniform_covariance[-2, -2] = poisson_v
                    self.uniform_covariance[-1, -1] = poisson_d
                    self.unmeasurable_state_mean = np.array([[poisson_mean_v], [poisson_mean_d]])
        else:
            self.uniform_radius = 0
            self.uniform_weight = 0
            self.uniform_adjust = 0
            self.uniform_angle = (0, 0)
            self.uniform_covariance = None
            self.unmeasurable_state_mean = None

        self.poisson_birth_state, self.poisson_birth_var = poisson_birth_state_models.get_model(
                                                                                model_name=poisson_states_model_name)

        if filter_class == 'Linear':
            self.filt = LinearFilter()
        elif filter_class == 'UKF':
            points = MerweScaledSigmaPoints(self.state_dims, alpha=ukf_alpha, beta=ukf_beta, kappa=ukf_kappa)
            self.filt = UKF(dt=dt, dim_x=self.state_dims, dim_z=len(self.measurement_model[1]), points=points,
                            fx=self.motion_model, hx=self.measurement_model)
        elif filter_class == 'Mixed':
            points = MerweScaledSigmaPoints(self.state_dims, alpha=ukf_alpha, beta=ukf_beta, kappa=ukf_kappa)
            self.filt = MixedFilter(dt=dt, dim_x=self.state_dims, dim_z=len(self.measurement_models['Bicycle'][1]),
                                    points=points, fx=self.motion_model, hx=self.measurement_models['Bicycle'])
        else:
            raise ValueError('Choose a filter that exists you fool!')

        if self.poisson_birth_state:
            assert np.shape(self.poisson_birth_state)[1] == self.state_dims, \
                'State dimension missmatch. Have: {}, Want: {}'.format(np.shape(self.poisson_birth_state)[1],
                                                                       self.state_dims)
