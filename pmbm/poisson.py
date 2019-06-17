import numpy as np
from math import atan2, sqrt
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import cdist
from utils.moment_matching import moment_matching_dists
from utils.constants import XLIM, ZLIM


class Distribution:
    def __init__(self, state, variance, weight, object_class, motion_model):
        assert state.shape[1] == 1, "Input state is not column vector"
        assert state.shape[0] == variance.shape[0], "State vector not aligned with Covariance matrix"
        self.state_dim = state.shape[0]
        self.state = state
        self.variance = variance
        self.weight = weight
        self.object_class = object_class
        self.motion_model = motion_model
        self.motion_noise = motion_model.get_Q(self.object_class)

    def predict(self, filt, survival_probability):
        if self.motion_model.model == 0:
            velocity = self.state[3]
        elif self.motion_model.model == 1:
            velocity = np.linalg.norm(self.state[2:4, :])
        else:
            velocity = 0
        if velocity != 0:
            _state, _variance = filt.predict(self.state, self.variance, self.motion_model, self.motion_noise, self.object_class)
            self.state = _state
            self.variance = _variance
        self.weight *= survival_probability

    def update(self, detection_probability):
        self.weight *= detection_probability

    def __repr__(self):
        return '<Distribution Class \n Weight: {} Object_class: {} \n'.format(self.weight, self.object_class)
        #return '<Distribution Class \n Weight: {} \n State: \n {} \n Variance: \n  {}> \n'.format(self.weight, self.state, self.variance)


class Poisson:
    """
    Class to hold all poisson distributions. Methods include birth, prediction, merge, prune, recycle.

    :param birth_state:         Where to birth new poisson distributions
    :param birth_var:           What covariances the new poissons should have.
    :param birth_weight_factor: Weight of the new distributions
    :param prune_threshold:     Which weight threshold for pruning
    :param merge_threshold:     Which distance threshold for merge of poisson distributions
    :param reduce_factor:       How much should the weight be reduced for each timestep
    :return:
    """

    def __init__(self,
                 birth_state,
                 birth_var,
                 birth_weight,
                 prune_threshold,
                 merge_threshold,
                 reduce_factor,
                 uniform_weight,
                 uniform_radius,
                 uniform_angle,
                 uniform_adjust,
                 state_dim):

        self.birth_weight = birth_weight
        if birth_state is None:
            self.number_of_births = 0
        else:
            self.number_of_births = len(birth_state)

        self.state_dim = state_dim
        self.distributions = []

        self.birth_state = birth_state
        self.birth_var = birth_var
        self.prune_threshold = prune_threshold
        self.merge_threshold = merge_threshold
        self.reduce_factor = reduce_factor

        self.uniform_radius = uniform_radius
        self.uniform_angle = uniform_angle  # Tuple with min, max in radius
        self.uniform_adjust = uniform_adjust
        self.uniform_weight = uniform_weight
        self.uniform_area = 0.5 * uniform_radius ** 2 * (uniform_angle[1] - uniform_angle[0])

        # Plot thingies
        self.x_range = [-50, 150]
        self.y_range = [-50, 150]
        self.window_min = np.array([self.x_range[0], self.y_range[0]])
        self.window_size = np.array([self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0]])
        self.window_intensity = np.prod(self.window_size)

        self.grid_length = [200, 200]
        self.x_dim = np.linspace(self.x_range[0], self.x_range[1], self.grid_length[0])
        self.y_dim = np.linspace(self.y_range[0], self.y_range[1], self.grid_length[1])
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_dim, self.y_dim)
        self.grid = np.dstack((self.x_mesh, self.y_mesh))
        self.intensity = np.zeros((self.grid_length[1], self.grid_length[0]))

    def __repr__(self):
        return '<Poisson Class \n Distributions: \n {}>'.format(self.distributions)

    def give_birth(self):
        for i in range(self.number_of_births):
            self.distributions.append(Distribution(self.birth_state[i], self.birth_var, self.birth_weight))

    def predict(self, filt, survival_probability):
        for dist in self.distributions:
            dist.predict(filt, survival_probability)
        self.give_birth()

    def update(self, detection_probability):
        for dist in self.distributions:
            dist.update(detection_probability)

    def prune(self):
        self.distributions[:] = [distribution for distribution in self.distributions if
                                 distribution.weight > self.prune_threshold]

    def within_uniform(self, point):
        angle = atan2(point[1] + self.uniform_adjust, point[0])
        radius = sqrt(point[0]**2 + point[1]**2)
        return self.uniform_angle[0] < angle < self.uniform_angle[1] and radius < self.uniform_radius

    def merge(self):
        _new_distributions = []
        while self.distributions:
            if len(self.distributions) == 1:
                _new_distributions.append(self.distributions[0])
                del self.distributions[0]

            else:
                self.distributions.sort(key=lambda x: x.weight, reverse=True)
                cdistr = self.distributions[0]
                states = np.array([distr.state.reshape(cdistr.state_dim, )
                                   for index, distr in enumerate(self.distributions)
                                   if (index != 0 and distr.object_class == cdistr.object_class)])
                indices = [index
                           for index, distr in enumerate(self.distributions)
                           if (index != 0 and distr.object_class == cdistr.object_class)]

                if len(states) > 0:
                    distance = list(cdist(cdistr.state.reshape(1, cdistr.state_dim),
                                          states.reshape(len(states), cdistr.state_dim), metric='mahalanobis',
                                          VI=np.linalg.inv(cdistr.variance))[0])

                    indices_within_threshold = [index_value for counter, index_value in enumerate(indices)
                                                if distance[counter] < self.merge_threshold]
                else:
                    indices_within_threshold = False

                if indices_within_threshold:
                    indices_within_threshold.append(0)
                    merge_list = [distr for index, distr in enumerate(self.distributions)
                                  if index in indices_within_threshold]
                    merge_weights = [distr.weight for index, distr in enumerate(self.distributions)
                                     if index in indices_within_threshold]

                    # Merge with the ones that are inside of threshold
                    _state, _variance, _weight = moment_matching_dists(merge_list, merge_weights)
                    _new_distributions.append(Distribution(state=_state,
                                                           variance=_variance,
                                                           weight=_weight,
                                                           object_class=cdistr.object_class,
                                                           motion_model=cdistr.motion_model))

                    # Remove the merged ones from the distribution list.
                    for i in sorted(indices_within_threshold, reverse=True):
                        del self.distributions[i]
                else:
                    _new_distributions.append(self.distributions[0])
                    del self.distributions[0]

        # When the old distribution list is empty, overwrite with new.
        self.distributions = _new_distributions

    def recycle(self, bernoulli, motion_model, object_class):
        # TODO: Change to existence instead of birth weight factor?
        _distr = Distribution(state=bernoulli.state,
                              variance=bernoulli.variance,
                              weight=bernoulli.existence,
                              object_class=object_class,
                              motion_model=motion_model)
        self.distributions.append(_distr)

    def reduce_weight(self, index):
        self.distributions[index].weight *= self.reduce_factor

    def plot(self, measurement_model):
        self.compute_intensity(measurement_model)
        plt.figure()
        for counter, distribution in enumerate(self.distributions):
            plt.plot(distribution.state[0], distribution.state[1], 'bo',
                     markersize=self.distributions[counter].weight * 10)
        plt.title("Poisson Distribution")
        plt.contourf(self.x_dim, self.y_dim, self.intensity, alpha=0.5)
        plt.xlim(XLIM[0], XLIM[1])
        plt.ylim(ZLIM[0], ZLIM[1])
        plt.show()

    def compute_intensity(self, measurement_model):
        self.reset_intensity()
        # increase intensity based on distributions in PPP
        for counter, distribution in enumerate(self.distributions):
            measurable_states = measurement_model @ distribution.state
            measurable_variance = measurement_model @ distribution.variance @ measurement_model.T
            _rv = stats.multivariate_normal(mean=measurable_states.reshape(np.shape(measurement_model)[0]),
                                            cov=measurable_variance)
            sampled_pdf = _rv.pdf(self.grid) * self.distributions[counter].weight
            self.intensity += sampled_pdf

    def reset_intensity(self):
        self.intensity = np.zeros((self.grid_length[1], self.grid_length[0]))
