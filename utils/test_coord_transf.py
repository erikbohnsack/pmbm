import unittest
import platform
from data_utils.kitti_stuff import Kitti, IMU
from .coord_transf import coordinate_transform_bicycle
import numpy as np
from math import pi, sqrt

if platform.system() == 'Darwin':
    root = '/Users/erikbohnsack/data'
else:
    root = '/home/mlt/data'


class TestCoordTransf(unittest.TestCase):
    def test_coord_transf(self):
        state = np.ones((5, 1))
        imud = IMU(0, 1, -1, 0)
        dt = 1
        transformed_state = coordinate_transform_bicycle(state, imud, dt)
        assert (transformed_state[0:2] == np.zeros((2, 1))).all()

    def test_coord_transf_angle(self):
        state = np.array([[2.], [2.], [0.], [0.], [0.]])
        imud = IMU(0, 0., 0., pi / 2)
        dt = 1
        transformed_state = coordinate_transform_bicycle(state, imud, dt)
        np.testing.assert_allclose(transformed_state[0:2], np.array([[2.], [-2.]]))

    def test_coord_trans_kitti(self):
        sequence_id = 0
        kitti = Kitti(root)
        imud = kitti.load_imu(sequence_id)
        kitti.lbls = kitti.load_labels(sequence_id)

        timesteps = 2
        TID = 0
        dt = 0.1
        for frame_idx in range(timesteps):
            labels = kitti.lbls[frame_idx]
            next_labels = kitti.lbls[frame_idx + 1]
            for label in labels:
                for next_label in next_labels:
                    if label.track_id == next_label.track_id == TID:

                        x = label.location[0]
                        z = label.location[2]
                        psi = - label.rotation_y
                        next_x = next_label.location[0]
                        next_z = next_label.location[0]
                        next_psi = - next_label.rotation_y

                        state = np.array([[x], [z], [psi], [0], [0]])
                        transformed_state = coordinate_transform_bicycle(state, imud[frame_idx], dt)
                        print('------------------')
                        print('Time: {}, state:\n {}'.format(frame_idx, state))
                        print('Time: {}, transformed state:\n {}'.format(frame_idx, transformed_state))
                    else:
                        continue

                #print("Frame: {} , rotation: {}".format(frame_idx, -label.rotation_y))




