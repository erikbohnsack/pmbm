import numpy as np
from .single_target_hypothesis import SingleTargetHypothesis
from scipy.spatial.distance import cdist


class Target:
    def __init__(self,
                 time_of_birth,
                 state,
                 variance,
                 weight,
                 existence,
                 measurement_index,
                 motion_model,
                 object_class,
                 survival_probability,
                 detection_probability,
                 measurement_model,
                 measurement_noise,
                 gating_distance,
                 verbose=False):

        self.single_target_hypotheses = {0: SingleTargetHypothesis(measurement_index=measurement_index,
                                                                   state=state,
                                                                   variance=variance,
                                                                   weight=weight,
                                                                   existence=existence,
                                                                   time_of_birth=time_of_birth)}
        self.next_single_id = 1
        self.time_of_birth = time_of_birth
        self.object_class = object_class
        self.motion_model = motion_model
        self.motion_noise = motion_model.get_Q(object_class)
        self.measurement_model = measurement_model
        self.measurement_noise = measurement_noise
        self.measurement_dims = np.shape(measurement_model)[0]
        self.survival_probability = survival_probability
        self.detection_probability = detection_probability
        self.gating_distance = gating_distance

        self.target_weight = None

        self.verbose = verbose

    def __repr__(self):
        return '<Target\n Birth:{}, nof_STH: {}, class: {}, weight: {}\n single_target_hypotheses: \n{}\n'.format(
            self.time_of_birth,
            len(self.single_target_hypotheses),
            self.object_class,
            round(self.target_weight,3),
            self.single_target_hypotheses)

    def coord_transform(self, coord_trans_fn, node, imu_data, dt):
        node.state = coord_trans_fn(node.state, imu_data, dt, self.object_class)

    def predict(self, filt, node):
        _state, _variance = filt.predict(state=node.state,
                                         variance=node.variance,
                                         motion_model=self.motion_model,
                                         motion_noise=self.motion_noise,
                                         object_class=self.object_class)
        node.state = _state
        node.variance = _variance

        node.existence *= self.survival_probability

    def update(self, measurements, classes, current_time, filt):
        """
        Input:  Class Target, new measurements and current time
           Output: The updated target
        :param measurements:
        :param classes:
        :param current_time:
        :param filt:
        :return:
        """
        measurements = [meas[0:self.measurement_dims] for meas in measurements]
        if self.time_of_birth != current_time:
            self.update_old(measurements, classes, current_time, filt)

    def update_old(self, measurements, classes, current_time, filt):
        output_nodes = {}
        for single_id, node in self.single_target_hypotheses.items():
            # print("Create new STH for {}".format(single_id))
            _children = []
            # Missed detection hypothesis
            missed_state = node.state
            missed_variance = node.variance

            # For numerical stability
            missed_variance = 0.5 * (missed_variance + missed_variance.transpose())
            missed_existence = node.existence * (1 - self.detection_probability) / (
                    1 - node.existence + node.existence * (1 - self.detection_probability))
            missed_weight = node.weight + np.log(1 - node.existence + node.existence * (1 - self.detection_probability))

            missed_detection_hypo = SingleTargetHypothesis(state=missed_state,
                                                           variance=missed_variance,
                                                           existence=missed_existence,
                                                           weight=missed_weight,
                                                           measurement_index=-1,
                                                           time_of_birth=current_time)

            output_nodes[self.next_single_id] = missed_detection_hypo
            # print("SHT {} should get child {}".format(single_id, self.next_single_id))
            _children.append(self.next_single_id)
            self.next_single_id += 1

            # If we don't get any measurements we are done with update step here!
            if len(measurements) == 0:
                self.single_target_hypotheses[single_id].children = _children
                continue

            for j, measurement in enumerate(measurements):
                # If current measurement is empty due to missed detection: go to next possible measurement
                if len(measurement) == 0:
                    continue

                # If the measurements is for another object class, continue to another measurement
                if not self.object_class == classes[j]:
                    continue

                measurable_states = self.measurement_model @ node.state
                measurable_cov = self.measurement_model @ node.variance @ self.measurement_model.transpose() + \
                                 self.measurement_noise
                distance = cdist(measurement.reshape(1, self.measurement_model.shape[0]),
                                 measurable_states.reshape(1, self.measurement_model.shape[0]), metric='mahalanobis',
                                 VI=np.linalg.inv(measurable_cov))

                if distance > self.gating_distance:
                    continue

                detected_state, detected_variance, meas_likelihood = filt.update(node.state, node.variance,
                                                                                 measurement,
                                                                                 self.measurement_model,
                                                                                 self.measurement_noise,
                                                                                 self.object_class)
                detected_existence = 1

                detected_weight = node.weight + np.log(node.existence * self.detection_probability * meas_likelihood)
                single_cost = detected_weight - missed_weight

                detected_node = SingleTargetHypothesis(measurement_index=j,
                                                       state=detected_state,
                                                       variance=detected_variance,
                                                       existence=detected_existence,
                                                       weight=detected_weight,
                                                       time_of_birth=current_time,
                                                       single_cost=single_cost)

                output_nodes[self.next_single_id] = detected_node
                _children.append(self.next_single_id)
                self.next_single_id += 1
            self.single_target_hypotheses[single_id].children = _children

        self.single_target_hypotheses.update(output_nodes)

    def remove_single_hypos(self, single_ids):
        for single_id in single_ids:
            del self.single_target_hypotheses[single_id]

