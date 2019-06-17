import numpy as np
import time
from .poisson import Poisson
from .global_hypothesis import GlobalHypothesis
from .target import Target

from utils.constants import LARGE
from utils.matrix_stuff import log_sum
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from utils.moment_matching import moment_matching
from murty import Murty
import copy


class PMBM:
    def __init__(self, config):
        if config.motion_model_name != 'Mixed':
            self.mixed = False
            self.measurement_model = config.measurement_model
            self.measurement_noise = config.measurement_noise
            self.meas_dims = config.measurement_model.shape[0]
            self.unmeasurable_state_mean = config.unmeasurable_state_mean
            self.uniform_covariance = config.uniform_covariance
        else:
            self.mixed = True
            self.measurement_models = config.measurement_models
            self.measurement_noises = config.measurement_noises
            self.meas_dims = 3
            self.unmeasurable_state_means = config.unmeasurable_state_means
            self.uniform_covariances = config.uniform_covariances

        self.state_dims = config.state_dims
        self.motion_model = config.motion_model

        self.survival_probability = config.survival_probability
        self.detection_probability = config.detection_probability
        self.gating_distance = config.gating_distance

        self.birth_gating_distance = config.birth_gating_distance

        self.poisson_birth_var = config.poisson_birth_var
        self.poisson = Poisson(birth_state=config.poisson_birth_state,
                               birth_var=config.poisson_birth_var,
                               prune_threshold=config.prune_threshold_poisson,
                               birth_weight=config.birth_weight,
                               reduce_factor=config.poisson_reduce_factor,
                               merge_threshold=config.poisson_merge_threshold,
                               uniform_weight=config.uniform_weight,
                               uniform_radius=config.uniform_radius,
                               uniform_angle=config.uniform_angle,
                               uniform_adjust=config.uniform_adjust,
                               state_dim=self.state_dims)
        self.filt = config.filt
        self.prune_threshold_targets = config.prune_threshold_targets
        self.prune_global_hypo = config.prune_threshold_global_hypo
        self.prune_single_existence = config.prune_single_existence

        self.desired_nof_global_hypos = config.desired_nof_global_hypos
        self.max_nof_global_hypos = config.max_nof_global_hypos
        self.min_new_nof_global_hypos = config.min_new_nof_global_hypos
        self.max_new_nof_global_hypos = config.max_new_nof_global_hypos
        self.global_init_weight = config.global_init_weight
        self.clutter_intensity = config.clutter_intensity

        self.global_hypotheses = {}
        self.global_hypotheses_counter = 0
        self.targets = {}
        self.new_targets = []
        self.target_counter = 0
        self.current_time = 0
        self.nof_new_targets = 0
        self.estimated_targets = None
        self.show_predictions = config.show_predictions
        self.verbose = False

        self.dt = config.dt
        self.coord_transform = config.coord_transform

    def __repr__(self):
        return '< PMBM Class \nCurrent time: {}\nNumber of targets: {}\nGliobalHypos: {} \n#Poissons: {} \n'.format(
            self.current_time,
            len(self.targets),
            self.global_hypotheses,
            len(self.poisson.distributions))

    def run(self, measurements, classes, imu_data, time_idx, verbose=False, verbose_time=False):
        if verbose: print('Targets Before prediction: \n{}'.format(self.targets))
        if verbose: print('GlobalHypos Before prediction: \n{}'.format(self.global_hypotheses))
        self.current_time = time_idx
        if verbose_time:
            tic = time.time()
            self.predict(imu_data)
            toc_predict = time.time() - tic
            tic = time.time()
            self.update(measurements, classes)
            toc_update = time.time() - tic
            tic = time.time()
            self.target_estimation()
            toc_estimation = time.time() - tic
            tic = time.time()
            self.reduction()
            toc_reduction = time.time() - tic
            print('\t\tT_Prediction:\t {}\tms\n'
                  '\t\tT_Update:\t {}\tms\n'
                  '\t\tT_Estimation\t {}\tms\n'
                  '\t\tT_Reduction\t {}\tms\n'.format(round(toc_predict * 1000, 1),
                                                      round(toc_update * 1000, 1),
                                                      round(toc_estimation * 1000, 1),
                                                      round(toc_reduction * 1000, 1)))
        else:
            self.predict(imu_data)
            self.update(measurements, classes)
            self.target_estimation()
            self.reduction()

        if verbose: print('Targets After update and reduction: \n{}'.format(self.targets))
        if verbose: print('GlobalHypos After update and reduction: \n{}'.format(self.global_hypotheses))

    def predict(self, imu_data):
        # For Poisson
        self.poisson.predict(filt=self.filt, survival_probability=self.survival_probability)

        # For all targets
        if self.targets:
            for target_id, target in self.targets.items():
                # For all single target hypothesis
                for single_id, hypo in target.single_target_hypotheses.items():
                    if self.coord_transform is not None:
                        target.coord_transform(self.coord_transform, hypo, imu_data, self.dt)
                    target.predict(self.filt, hypo)

    def update(self, measurements, classes):
        # Check if any Poisson should be birth, and in that case give birth
        self.possible_new_targets(measurements, classes)

        # Update current tracks
        for target_id, target in self.targets.items():
            target.update(measurements, classes, self.current_time, self.filt)
        self.poisson.update(self.detection_probability)

        # Global hypotheses update and normalize weights
        self.update_global_hypotheses(measurements)
        self.normalize_global_hypotheses()

    def update_global_hypotheses(self, measurements):
        """
        Updates global hypotheses. If there is no global hypo since before, it creates a new one.

        If there are previous global hypothesis it creates new hypotheses using Murty's algorithm.
        :param measurements:
        :return:
        """

        # If no global hypo, but targets
        # This is cool because if we don't have any global hypotheses that means that all self.targets are among
        # new_targets and thus must have one and only one measurement connected to it
        if not self.global_hypotheses and self.targets:
            hypo = []
            for target_id, target in self.targets.items():
                if len(target.single_target_hypotheses) == 1:
                    hypo.append((target_id, 0))
                else:
                    raise ValueError("More than one single hypo in new target")
            self.global_hypotheses[self.global_hypotheses_counter] = GlobalHypothesis(self.global_init_weight, hypo)
            self.global_hypotheses_counter += 1
        else:
            new_global_hypos = {}
            for global_hypo_id, global_hypo in self.global_hypotheses.items():
                cost_matrix, row_col_2_tid_sid, missed_meas_col_2_tid_sid = self.create_cost(global_hypo, measurements)

                new_hypos = self.generate_new_global_hypos(cost_matrix,
                                                           global_hypo,
                                                           row_col_2_tid_sid,
                                                           missed_meas_col_2_tid_sid)
                for hypo in new_hypos:
                    new_global_hypos[self.global_hypotheses_counter] = hypo
                    self.global_hypotheses_counter += 1
            self.global_hypotheses = new_global_hypos

    def create_cost(self, global_hypo, measurements):
        """
        Creates a cost matrix for a specific global hypothesis.

        :param global_hypo:
        :param measurements:
        :return:
        """
        if self.nof_new_targets > 0:
            W_nt, new_target_map, new_measurement_map_meas2row, new_measurement_map_row2meas, nt_row_col_2_tid_sid = \
                self.new_targets_cost(measurements)
            W_nt_flag = True
        else:
            W_nt = None
            new_measurement_map_meas2row = None
            new_measurement_map_row2meas = None
            nt_row_col_2_tid_sid = None
            W_nt_flag = False

        nof_old_targets = len(self.targets) - self.nof_new_targets
        if nof_old_targets > 0:
            W_ot, old_target_map, old_measurement_map_row2meas, ot_row_col_2_tid_sid, missed_meas_target_2_tid_sid = \
                self.old_targets_cost(global_hypo, measurements, new_measurement_map_meas2row)
            W_ot_flag = True
        else:
            W_ot = None
            old_measurement_map_row2meas = None
            W_ot_flag = False
            ot_row_col_2_tid_sid, missed_meas_target_2_tid_sid = None, {}

        if W_nt_flag and W_ot_flag:
            cost_matrix = np.hstack((W_ot, W_nt))
            row_col_2_tid_sid = np.hstack((ot_row_col_2_tid_sid, nt_row_col_2_tid_sid))
            measurement_map_row2meas = {**old_measurement_map_row2meas, **new_measurement_map_row2meas}

        elif W_nt_flag:
            cost_matrix = W_nt
            measurement_map_row2meas = new_measurement_map_row2meas
            row_col_2_tid_sid = nt_row_col_2_tid_sid
        elif W_ot_flag:
            cost_matrix = W_ot
            measurement_map_row2meas = old_measurement_map_row2meas
            row_col_2_tid_sid = ot_row_col_2_tid_sid
        else:
            cost_matrix = None
            measurement_map_row2meas = None
            row_col_2_tid_sid = None

        nof_rows_mapped = len(measurement_map_row2meas)
        reduced_cost_matrix = cost_matrix[:nof_rows_mapped, :]
        return reduced_cost_matrix, row_col_2_tid_sid, missed_meas_target_2_tid_sid

    def new_targets_cost(self, measurements):
        nof_measurements = len(measurements)
        W_nt = np.full((nof_measurements, len(self.new_targets)), LARGE)

        row_col_2_tid_sid = np.full((nof_measurements, len(self.new_targets), 2), [None, None])

        new_target_map = {}
        new_measurement_map_meas2row = {}
        new_measurement_map_row2meas = {}
        for counter, target_id in enumerate(self.new_targets):
            assert len(self.targets[
                           target_id].single_target_hypotheses) == 1, "Multiple single target hypotheses in new target. That is bad. "
            assert self.targets[
                       target_id].time_of_birth == self.current_time, "The new target wasn't created at this time step. Hence it shouldn't be here... "
            assert self.targets[target_id].single_target_hypotheses[
                       0].measurement_index != -1, "Must have a measurement connected to this target!"

            measurement_index = self.targets[target_id].single_target_hypotheses[0].measurement_index
            _weight = self.targets[target_id].single_target_hypotheses[0].weight
            new_target_map[counter] = target_id
            new_measurement_map_meas2row[measurement_index] = counter
            new_measurement_map_row2meas[counter] = measurement_index
            W_nt[counter, counter] = -_weight
            row_col_2_tid_sid[counter, counter] = [target_id, 0]
        return W_nt, new_target_map, new_measurement_map_meas2row, new_measurement_map_row2meas, row_col_2_tid_sid

    def old_targets_cost(self, global_hypo, measurements, new_measurement_map_meas2row):
        nof_measurements = len(measurements)
        nof_old_targets = len(global_hypo.hypothesis)
        W_ot = np.full((nof_measurements, nof_old_targets), LARGE)

        row_col_2_tid_sid = np.full((nof_measurements, nof_old_targets, 2), [None, None])
        missed_meas_target_2_tid_sid = {}

        old_target_map = {}
        old_measurement_map_meas2row = {}
        old_measurement_map_row2meas = {}

        if not (new_measurement_map_meas2row is None):
            row_counter = max(new_measurement_map_meas2row.values()) + 1
        else:
            new_measurement_map_meas2row = {}
            row_counter = 0

        for index, id_combo in enumerate(global_hypo.hypothesis):
            old_target_map[index] = id_combo[0]
            for child in self.targets[id_combo[0]].single_target_hypotheses[id_combo[1]].children:
                _meas_index = self.targets[id_combo[0]].single_target_hypotheses[child].measurement_index
                if _meas_index == -1:
                    missed_meas_target_2_tid_sid[id_combo[0]] = [id_combo[0], child]
                    continue
                if _meas_index in new_measurement_map_meas2row.keys():
                    row = new_measurement_map_meas2row[_meas_index]
                elif _meas_index in old_measurement_map_meas2row.keys():
                    row = old_measurement_map_meas2row[_meas_index]
                else:
                    row = row_counter
                    old_measurement_map_meas2row[_meas_index] = row_counter
                    old_measurement_map_row2meas[row_counter] = _meas_index
                    row_counter += 1
                row_col_2_tid_sid[row, index] = [id_combo[0], child]
                _cost = self.targets[id_combo[0]].single_target_hypotheses[child].single_cost
                assert _cost != LARGE, 'Cost is LARGE, did you add a cost for this STH?'
                W_ot[row, index] = - _cost

        return W_ot, old_target_map, old_measurement_map_row2meas, row_col_2_tid_sid, missed_meas_target_2_tid_sid

    def generate_new_global_hypos(self, cost_matrix, global_hypo, row_col_2_tid_sid, missed_meas_target_2_tid_sid):
        """
        Generates new global hypotheses for a specific global hypo with respective cost matrix.
        Uses Murty's algorithm to generate k different assignments of the measurements.

        :param cost_matrix:
        :param global_hypo:
        :param row_col_2_tid_sid:
        :param missed_meas_target_2_tid_sid:
        :return:
        """
        if (len(cost_matrix) == 0) or (cost_matrix is None):
            return []

        if global_hypo.weight > LARGE:
            global_hypo.weight = LARGE
        _old_weight = global_hypo.weight

        k = int(np.ceil(self.desired_nof_global_hypos * np.exp(global_hypo.weight)))
        k = max(min(k, self.max_new_nof_global_hypos), self.min_new_nof_global_hypos)
        mgen = Murty(cost_matrix)
        new_global_hypos = []
        old_targets_in_current_hypo = [x[0] for x in global_hypo.hypothesis]

        for iteration in range(k):
            new_global_hypo = []
            ok, cost, solution = mgen.draw()
            sol = solution.tolist()

            if (not ok) or (cost >= 0.8 * LARGE):
                break

            old_targets_in_current_solution = []

            for measurement_row, target_column in enumerate(sol):
                target_id, single_id = row_col_2_tid_sid[measurement_row, target_column]
                # Add the new target
                new_global_hypo.append((target_id, single_id))
                if target_id not in self.new_targets:
                    old_targets_in_current_solution.append(target_id)

            missed_old_targets = [x for x in old_targets_in_current_hypo if x not in old_targets_in_current_solution]
            for mot in missed_old_targets:
                m_tid, m_sid = missed_meas_target_2_tid_sid[mot]
                if self.targets[m_tid].single_target_hypotheses[m_sid].existence > self.prune_single_existence:
                    new_global_hypo.append((m_tid, m_sid))
            _weight = _old_weight - cost

            if new_global_hypo:
                new_global_hypo = list(set(new_global_hypo))
                new_global_hypos.append(GlobalHypothesis(_weight, new_global_hypo))
        return new_global_hypos

    def get_global_weight(self, hypo):
        _weight_array = np.array([])
        for target_id, single_id in hypo:
            _weight_array = np.append(_weight_array, self.targets[target_id].single_target_hypotheses[single_id].weight)
        if len(_weight_array) > 0:
            weight_sum = log_sum(_weight_array)
        else:
            weight_sum = - LARGE
        return weight_sum

    def normalize_global_hypotheses(self):
        if len(self.global_hypotheses) == 0:
            return
        _weights = np.array([h.weight for key, h in self.global_hypotheses.items()])
        _sum = log_sum(_weights)
        for key, h in self.global_hypotheses.items():
            h.weight -= _sum

    def reduction(self):
        # Prune global hypothesis
        self.prune_global()
        self.normalize_global_hypotheses()

        # Cap number of global hypos
        if len(self.global_hypotheses) > self.max_nof_global_hypos:
            self.cap_global_hypos()
            self.normalize_global_hypotheses()

        # Remove unused Single Target Hypotheses
        self.remove_unused_STH()

        # Recycle remaining Bernoulli components
        self.recycle_targets()
        self.normalize_global_hypotheses()

        # Merge Global hypotheses that are duplicates
        self.merge_global_duplicates()
        self.normalize_global_hypotheses()

        # Prune and merge poissons
        self.poisson.prune()
        self.poisson.merge()

    def prune_global(self):
        global_hypos_to_remove = [global_id for global_id, global_hypo in self.global_hypotheses.items()
                                  if global_hypo.weight < self.prune_global_hypo]

        # Remove global hypos
        for key in global_hypos_to_remove:
            # print('Prune_global: removing key: {}'.format(key))
            # print('Global hypo: {}'.format(self.global_hypotheses[key]))
            self.global_hypotheses.pop(key)

    def recycle_targets(self):
        """
        For each target, sum over all global hypothesis each single target hypothesis is a part of. The terms of the sum
        is the global hypo weight times the single target hypothesis probability of existence
        :return:
        """
        keys_to_remove = []
        # print('Recycle targets, hypos: {}'.format(self.global_hypotheses))
        for target_id, target in self.targets.items():
            weight_array = np.array([])
            sids_to_remove = []
            for single_id, single_hypo in target.single_target_hypotheses.items():
                sid_relevant = False
                for global_id, hypo in self.global_hypotheses.items():
                    if (target_id, single_id) in hypo.hypothesis:
                        weight_array = np.append(weight_array, np.log(single_hypo.existence) + hypo.weight)
                        sid_relevant = True
                if not sid_relevant:
                    sids_to_remove.append(single_id)
                    # print('Recycling: (TID, SID) = ({}, {})'.format(target_id, single_id))
                    # print('Target: \n{}'.format(self.targets[target_id]))
            [target.single_target_hypotheses.pop(sid) for sid in sids_to_remove]

            if len(weight_array) > 0:
                weighted_sum = log_sum(weight_array)
            else:
                weighted_sum = - LARGE

            target.target_weight = weighted_sum  # Just to being able to look at it at a later stage

            if weighted_sum < self.prune_threshold_targets:
                # recycle all single target hypos in target
                for single_id, single_hypo in self.targets[target_id].single_target_hypotheses.items():
                    self.poisson.recycle(bernoulli=single_hypo,
                                         motion_model=self.targets[target_id].motion_model,
                                         object_class=self.targets[target_id].object_class)
                # print('Removing Target TID: {}'.format(target_id))
                # print('Target: \n{}'.format(self.targets[target_id]))
                keys_to_remove.append(target_id)

        # Remove the targets after recycling
        [self.targets.pop(key) for key in keys_to_remove]

        # Remove targets from global hypotheses
        globals_to_remove = []
        for key, global_hypo in self.global_hypotheses.items():
            new_hypothesis = [hypo for hypo in global_hypo.hypothesis if hypo[0] not in keys_to_remove]
            global_hypo.hypothesis = new_hypothesis

            if len(new_hypothesis) == 0:
                globals_to_remove.append(key)

        # Remove empty global hypos
        [self.global_hypotheses.pop(key) for key in globals_to_remove]

    def remove_unused_STH(self):
        for tid, target in self.targets.items():
            keys_to_remove = [sid for sid, sth in target.single_target_hypotheses.items() if
                              sth.time_of_birth < self.current_time]
            [target.single_target_hypotheses.pop(key) for key in keys_to_remove]

    def cap_global_hypos(self):
        _tuples = [(key, global_hypo.weight) for key, global_hypo in self.global_hypotheses.items()]
        _tuples.sort(key=lambda x: x[1], reverse=True)

        _to_remove = _tuples[self.max_nof_global_hypos:]
        for _rmv in _to_remove:
            self.global_hypotheses.pop(_rmv[0])

    def merge_global_duplicates(self):
        " Merging global hypotheses that are the same. The weights are summed"
        new_globi = {}
        duplicate_keys = []
        for key, value in self.global_hypotheses.items():
            duplicate_keys.append([k for k, v in self.global_hypotheses.items() if v.hypothesis == value.hypothesis])
        duplicate_keys = [list(x) for x in set(tuple(x) for x in duplicate_keys)]

        for dup_keys in duplicate_keys:
            _key = dup_keys[0]
            _hypo = self.global_hypotheses[_key].hypothesis
            _weights = np.array([self.global_hypotheses[k].weight for k in dup_keys])
            _sum = log_sum(_weights)
            new_globi[_key] = GlobalHypothesis(_sum, _hypo)
        self.global_hypotheses = new_globi

    def gated_new_targets(self, meas, object_class):
        states_within_gate = []
        variances_within_gate = []
        weight_within_gate = np.array([])
        for ix, distribution in enumerate(self.poisson.distributions):
            # Gate class
            if distribution.object_class != object_class:
                continue
            meas_dims = np.shape(self.measurement_model)[0]
            # Perform measurement gating
            measurable_states = self.measurement_model @ distribution.state
            measurable_cov = self.measurement_model @ distribution.variance @ self.measurement_model.transpose() + \
                             self.measurement_noise
            distance = cdist(meas.reshape(1, meas_dims),
                             measurable_states.reshape(1, meas_dims), metric='mahalanobis',
                             VI=np.linalg.inv(measurable_cov))
            if distance > self.birth_gating_distance:
                if self.verbose: print(
                    'Gating occured at: state=\n {} \n with measurement= \n {}'.format(measurable_states, meas))
                continue

            states_within_gate.append(distribution.state)
            variances_within_gate.append(distribution.variance)
            weight_within_gate = np.append(weight_within_gate, distribution.weight)

            # Update the Poisson weight to compensate for also being a Bernoulli
            self.poisson.reduce_weight(index=ix)

        return states_within_gate, variances_within_gate, weight_within_gate

    def sum_likelihood(self, states, variances, weights, measurement):
        _sum = 0
        for ix, state in enumerate(states):
            innov_cov = self.measurement_model @ variances[
                ix] @ self.measurement_model.transpose() + self.measurement_noise
            innov_cov = 0.5 * (innov_cov + innov_cov.transpose())
            reshaped_meas = measurement.reshape(self.meas_dims, )
            reshaped_state = (self.measurement_model @ state).reshape(self.meas_dims, )
            _sum += weights[ix] * multivariate_normal.pdf(reshaped_meas, reshaped_state, innov_cov)
        return _sum

    def possible_new_targets(self, measurements, classes):
        self.nof_new_targets = 0
        self.new_targets = []

        for jx, meas in enumerate(measurements):
            if len(meas) == 0:
                continue
            object_class = classes[jx]

            uniform_bool = self.poisson.within_uniform(meas)

            if self.mixed:
                if object_class == 'Pedestrian' or object_class == 'Misc' or object_class == 'Person':
                    self.measurement_model = self.measurement_models['CV']
                    self.measurement_noise = self.measurement_noises['CV']
                    meas = meas[0:2]
                    self.meas_dims = 2
                    # Create target from measurement
                    meas_state = np.vstack((meas, self.unmeasurable_state_means['CV']))
                    meas_cov = self.uniform_covariances['CV']
                else:
                    self.measurement_model = self.measurement_models['Bicycle']
                    self.measurement_noise = self.measurement_noises['Bicycle']
                    self.meas_dims = 3
                    # Create target from measurement
                    meas_state = np.vstack((meas, self.unmeasurable_state_means['Bicycle']))
                    meas_cov = self.uniform_covariances['Bicycle']
            else:
                # Create target from measurement
                meas_state = np.vstack((meas, self.unmeasurable_state_mean))
                meas_cov = self.uniform_covariance

            # Gate with respect to poisson gaussian distributions.
            states_within_gate, variances_within_gate, weights_within_gate = self.gated_new_targets(meas, classes[jx])

            # if measurement not in uniform or within gates of poisson components, then continue.
            if not uniform_bool and not states_within_gate:
                continue

            # compute _e differently based on if vicinity of poisson componentes and/or in uniform
            if states_within_gate:
                if uniform_bool:
                    _e = self.detection_probability * self.poisson.uniform_weight / self.poisson.uniform_area + \
                         self.detection_probability * self.sum_likelihood(states_within_gate,
                                                                          variances_within_gate,
                                                                          weights_within_gate, meas)
                else:
                    _e = self.detection_probability * self.sum_likelihood(states_within_gate,
                                                                          variances_within_gate,
                                                                          weights_within_gate, meas)

                # Kalman update each poisson
                updated_states = []
                updated_variances = []
                updated_weights = []
                for ix, state in enumerate(states_within_gate):
                    _state, _variance, meas_likelihood = self.filt.update(state,
                                                                          variances_within_gate[ix],
                                                                          meas,
                                                                          self.measurement_model,
                                                                          self.measurement_noise,
                                                                          classes[jx])
                    updated_states.append(_state)
                    updated_variances.append(_variance)
                    updated_weights.append(
                        np.log(weights_within_gate[ix] * self.detection_probability * meas_likelihood))

                # Merge poissons by moment matching after kalman update.
                poisson_state, poisson_variance, poisson_weight = moment_matching(updated_states,
                                                                                  updated_variances,
                                                                                  updated_weights)
                # if uniform and poisson, then moment match the updated and the uniform
                if uniform_bool:
                    target_state, target_variance, _ = moment_matching([poisson_state, meas_state],
                                                                       [poisson_variance, meas_cov],
                                                                       [poisson_weight,
                                                                        self.poisson.uniform_weight /
                                                                        self.poisson.uniform_area])
                else:
                    target_state = poisson_state
                    target_variance = poisson_variance

            # If no poisson components, then just go with the uniform measurement state.
            else:
                _e = self.detection_probability * self.poisson.uniform_weight / self.poisson.uniform_area
                target_state = meas_state
                target_variance = meas_cov

            weight = np.log(_e + self.clutter_intensity)
            existence = _e / (_e + self.clutter_intensity)

            self.targets[self.target_counter] = Target(measurement_index=jx,
                                                       time_of_birth=self.current_time,
                                                       state=target_state,
                                                       variance=target_variance,
                                                       weight=weight,
                                                       existence=existence,
                                                       motion_model=self.motion_model,
                                                       survival_probability=self.survival_probability,
                                                       detection_probability=self.detection_probability,
                                                       measurement_model=self.measurement_model,
                                                       measurement_noise=self.measurement_noise,
                                                       gating_distance=self.gating_distance,
                                                       object_class=classes[jx])

            self.new_targets.append(self.target_counter)
            self.target_counter += 1
            self.nof_new_targets += 1

    def target_estimation(self):
        '''
        Decide which global hypotheses to use and output the targets of that global hypothese
        '''
        best_global_hypothese_key = -1
        best_weight = -LARGE
        for key, global_hypo in self.global_hypotheses.items():
            _weight = global_hypo.weight
            if _weight > best_weight:
                best_global_hypothese_key = key
                best_weight = _weight

        if best_global_hypothese_key == -1:
            self.estimated_targets = []
            return

        best_global_hypothese = self.global_hypotheses[best_global_hypothese_key]

        output_targets = []
        for _target_hypo in best_global_hypothese.hypothesis:
            target_idx = _target_hypo[0]
            single_hypo_idx = _target_hypo[1]
            if self.targets[target_idx].single_target_hypotheses[
                single_hypo_idx].existence > self.prune_threshold_targets:
                _temp = {}
                _temp['target_idx'] = target_idx
                _temp['single_hypo_idx'] = single_hypo_idx
                _temp['single_target'] = self.targets[target_idx].single_target_hypotheses[single_hypo_idx]
                _temp['state_predictions'] = []
                _temp['var_predictions'] = []
                _temp['object_class'] = self.targets[target_idx].object_class

                if self.show_predictions is not None:
                    _es = copy.deepcopy(self.targets[target_idx].single_target_hypotheses[single_hypo_idx].state)
                    _ev = copy.deepcopy(self.targets[target_idx].single_target_hypotheses[single_hypo_idx].variance)
                    _mm = copy.deepcopy(self.targets[target_idx].motion_model)
                    _mn = copy.deepcopy(self.targets[target_idx].motion_noise)
                    _oc = copy.deepcopy(self.targets[target_idx].object_class)
                    for i in range(self.show_predictions):
                        _es, _ev = self.filt.predict(state=_es,
                                                     variance=_ev,
                                                     motion_model=_mm,
                                                     motion_noise=_mn,
                                                     object_class=_oc)
                        _temp['state_predictions'].append(_es)
                        _temp['var_predictions'].append(_ev)
                output_targets.append(_temp)

        self.estimated_targets = output_targets
