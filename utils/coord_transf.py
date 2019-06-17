import math
import numpy as np
from math import atan2, sqrt


def coordinate_transform_bicycle(state, imu_data, dt, object_class=None):
    assert state.shape == (5, 1)
    # Euler integrate angle of ego vehicle
    angle = - imu_data.ru * dt

    # minus angle holds here as well
    rotation_matrix = np.array([[math.cos(angle), - math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
    translation = np.array([[- imu_data.vl * dt],
                            [imu_data.vf * dt]])

    output_state = np.copy(state)
    output_state[0:2] = rotation_matrix @ state[0:2] - translation
    output_state[2] += angle

    # same argument for minus translation here as above.
    return output_state


def coordinate_transform_CV(state, imu_data, dt, object_class=None):
    assert state.shape == (4, 1)
    # Minus as the imu_data is for the ego vehicle
    angle = - imu_data.ru * dt

    # minus angle holds here as well
    rotation_matrix = np.array([[math.cos(angle), - math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
    translation = np.array([[- imu_data.vl * dt],
                            [imu_data.vf * dt]])

    output_state = np.copy(state)
    output_state[0:2] = rotation_matrix @ state[0:2] - translation
    output_state[2:4] = rotation_matrix @ state[2:4]
    # same argument for minus translation here as above.
    return output_state


def coordinate_transform_CA(state, imu_data, dt, object_class=None):
    assert state.shape == (6, 1)
    # Minus as the imu_data is for the ego vehicle
    angle = - imu_data.ru * dt

    # minus angle holds here as well
    rotation_matrix = np.array([[math.cos(angle), - math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
    translation = np.array([[- imu_data.vl * dt],
                            [imu_data.vf * dt]])

    output_state = np.copy(state)
    output_state[0:2] = rotation_matrix @ state[0:2] - translation
    output_state[2:4] = rotation_matrix @ state[2:4]
    output_state[4:6] = rotation_matrix @ state[4:6]
    # same argument for minus translation here as above.
    return output_state



def coordinate_transform_mixed(state, imu_data, dt, object_class):
    if object_class == 'Pedestrian' or object_class == 'Misc' or object_class == 'Person':
        output_state = coordinate_transform_CV(state, imu_data, dt)
    else:
        output_state = coordinate_transform_bicycle(state, imu_data, dt)

    return output_state

def within_fov(point, min_angle=0.78, max_angle=2.45, max_radius=100):
    angle = atan2(point[1], point[0])
    radius = sqrt(point[0] ** 2 + point[1] ** 2)
    return min_angle < angle < max_angle and radius < max_radius
