import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from .dataset_stuff import Label, IMU

PossibleClasses = ['Car', 'Pedestrian', 'Van', 'Cyclist']

class Kitti:
    def __init__(self, ROOT, split='training'):
        self.path = os.path.join(ROOT, split)
        self.label_path = os.path.join(self.path, 'label_2')
        self.image_path = os.path.join(self.path, 'image_2')
        self.velo_path = os.path.join(self.path, 'velodyne')
        self.imu_path = os.path.join(self.path, 'oxts')
        self.measurements = None
        self.classes = None
        self.imgs, self.pcds, self.lbls, self.imus = None, None, None, None
        self.max_frame_idx = None

        self.dT = 0.1 # 10 Hz sampling rate for kitti tracking dataset

    def __repr__(self):
        return '<Kitti | Path: {}>'.format(self.path)

    def load(self, sequence_idx, frame_idx):
        self.imgs = self.load_image(sequence_idx, frame_idx)
        self.pcds = self.load_velos(sequence_idx)
        self.lbls = self.load_labels(sequence_idx)
        self.imus = self.load_imu(sequence_idx)

    def load_image(self, sequence_idx, frame_idx):
        file_path = self.image_path + '/' + str(sequence_idx).zfill(4) + '/' + str(frame_idx).zfill(6) + '.png'
        assert os.path.exists(file_path), 'File does not exist: {}'.format(file_path)
        image = cv2.imread(file_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_velos(self, sequence_idx):
        return None

    def load_labels(self, sequence_idx):
        file_path = os.path.join(self.label_path, str(sequence_idx).zfill(4) + '.txt')
        assert os.path.exists(file_path), 'File does not exist: {}'.format(file_path)
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        labels = []
        for line in lines:
            l = line.split(' ')
            lbl = Label(l[0],
                        l[1],
                        l[2],
                        l[3],
                        l[4],
                        l[5],
                        l[6:10],
                        l[10:13],
                        l[13:16],
                        l[16])
            labels.append(lbl)
        max_frame_idx = labels[-1].frame
        ld = {i: [] for i in range(max_frame_idx+1)}
        {l.frame: ld[l.frame].append(l) for l in labels if l.type[0] != 'DontCare'}
        self.max_frame_idx = max_frame_idx
        return ld

    def load_imu(self, sequence_idx):
        file_path = os.path.join(self.imu_path, str(sequence_idx).zfill(4) + '.txt')
        assert os.path.exists(file_path), 'File does not exist: {}'.format(file_path)
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()

        imud = {i: [] for i in range(len(lines))}

        for idx, line in enumerate(lines):
            l = line.split(' ')
            _vf = float(l[8])  # Velocity forward
            _vl = float(l[9])  # Velocity leftward
            _vu = float(l[22]) # Rotation around upward
            imud[idx] = IMU(idx, _vf, _vl, _vu)

        return imud


    def load_measurements(self, p_missed=0.05, p_clutter=0.05, sigma_xy=0.1, sigma_psi=0.2, radius_tuple=(0, 100),
                          angle_tuple=(0.78, 2.35)):
        assert self.lbls is not None, 'Labels not loaded'

        measurement_dict = {}
        class_dict = {}

        for frame_idx in range(self.max_frame_idx):
            true_states = [np.array([[m.location[0]], [m.location[2]], [m.rotation_y]])
                           for m in self.lbls[frame_idx]
                           if not m.type[0] == 'DontCare']

            true_class = [m.type[0] for m in self.lbls[frame_idx] if not m.type[0] == 'DontCare']

            measurements = []
            classes = []
            for ix, state in enumerate(true_states):
                if p_missed < random.random():
                    state[0] += random.normalvariate(0, sigma_xy)
                    state[1] += random.normalvariate(0, sigma_xy)
                    state[2] += random.normalvariate(0, sigma_psi)
                    measurements.append(state)
                    classes.append(true_class[ix])
                if p_clutter > random.random():
                    r = random.uniform(radius_tuple[0], radius_tuple[1])
                    angle = random.uniform(angle_tuple[0], angle_tuple[1])
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    s = np.array([[x], [y], [angle]])
                    measurements.append(s)
                    classes.append(random.choice(PossibleClasses))

            measurement_dict[frame_idx] = measurements
            class_dict[frame_idx] = classes

        self.measurements = measurement_dict
        self.classes = class_dict

    def get_measurements(self, frame_idx, measurement_dims, classes_to_track=['all']):
        assert self.lbls is not None, 'Labels not loaded'
        assert self.measurements is not None, 'Measurements not loaded'
        assert self.classes is not None, 'Classes not loaded'
        # Lazy coding yes indeed
        output_meas = []
        output_cls = []
        for i, cls in enumerate(self.classes[frame_idx]):
            if cls not in classes_to_track and 'all' not in classes_to_track :
                continue
            if measurement_dims == 3:
                meas = self.measurements[frame_idx][i]
            elif measurement_dims == 2:
                m = self.measurements[frame_idx][i]
                meas = np.array([m[0], m[1]])
            else:
                raise ValueError('Add what should happen for measurement_dims = {} and try again...'.format(measurement_dims))
            output_meas.append(meas)
            output_cls.append(cls)
        return output_meas, output_cls

    def get_measurements_old(self, frame_idx, measurement_dims, p_missed=0.05, p_clutter=0.05, sigma_xy=0.1, sigma_psi=0.2,
                         radius_tuple=(0, 100), angle_tuple=(0.78, 2.35)):
        assert self.lbls is not None, 'Labels not loaded'
        if measurement_dims == 3:
            true_states = [np.array([[m.location[0]], [m.location[2]], [m.rotation_y]]) for m in self.lbls[frame_idx] if not m.type[0]=='DontCare']
        elif measurement_dims == 2:
            true_states = [np.array([[m.location[0]], [m.location[2]]]) for m in self.lbls[frame_idx] if not m.type[0]=='DontCare']
        else:
            raise ValueError('Add what should happen for measurement_dims = {} and try again...'.format(measurement_dims))

        true_class = [m.type[0] for m in self.lbls[frame_idx] if not m.type[0]=='DontCare']
        measurements = []
        classes = []

        for ix, state in enumerate(true_states):
            if p_missed < random.random():
                state[0] += random.normalvariate(0, sigma_xy)
                state[1] += random.normalvariate(0, sigma_xy)
                if measurement_dims == 3:
                    state[2] += random.normalvariate(0, sigma_psi)
                measurements.append(state)
                classes.append(true_class[ix])
            if p_clutter > random.random():
                r = random.uniform(radius_tuple[0], radius_tuple[1])
                angle = random.uniform(angle_tuple[0], angle_tuple[1])
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                if measurement_dims == 3:
                    s = np.array([[x], [y], [angle]])
                else:
                    s = np.array([[x], [y]])
                #print("r: {}\n, angle: {}\n, clutter: {}".format(r, angle, s))
                measurements.append(s)
                classes.append(random.choice(PossibleClasses))
        return measurements, classes

    def get_bev_states(self, frame_idx, classes_to_track=['all']):
        assert self.lbls is not None, 'Labels not loaded'
        if 'all' in classes_to_track:
            return [np.array([[m.location[0]], [m.location[2]]]) for m in self.lbls[frame_idx] if not m.type[0]=='DontCare']
        else:
            return [np.array([[m.location[0]], [m.location[2]]]) for m in self.lbls[frame_idx] if m.type[0] in classes_to_track]

    def get_ego_bev_velocity(self, frame_idx):
        assert self.imus is not None, 'IMU data not loaded'
        return np.array([[-self.imus[frame_idx].vl], [self.imus[frame_idx].vf]])

    def get_ego_bev_rotation(self, frame_idx):
        assert self.imus is not None, 'IMU data not loaded'
        _r = self.dT * self.imus[frame_idx].ru
        return np.array([[np.cos(_r), -np.sin(_r)],[np.sin(_r), np.cos(_r)]])

    def get_ext_bev_rotation(self, frame_idx):
        assert self.imus is not None, 'IMU data not loaded'
        _r = - self.dT * self.imus[frame_idx].ru
        return np.array([[np.cos(_r), -np.sin(_r)],[np.sin(_r), np.cos(_r)]])

    def get_ego_bev_motion(self):
        assert self.imus is not None, 'IMU data not loaded'
        ego_state_history = np.array([], dtype=np.float).reshape(2,0)
        ego_state = np.zeros((2,1))
        for frame_idx in range(self.max_frame_idx+1):
            current_velocity = self.get_ego_bev_velocity(frame_idx)
            current_rotation = self.get_ego_bev_rotation(frame_idx)
            # TODO: test if better to rotate first and then take step
            ego_state = current_rotation @ ego_state + self.dT * current_velocity
            ego_state_history = np.hstack((ego_state_history, ego_state))
        return ego_state_history

    def get_bev_gt_trajectories(self, up_to_frame_idx):
        assert self.imus is not None or self.lbls is not None, 'Pick sequence and load data first.'
        states_to_plot = []
        for frame_idx in range(0, up_to_frame_idx):
            current_state = self.get_bev_states(frame_idx)
            current_velocity = self.get_ego_bev_velocity(frame_idx)
            current_rotation = self.get_ext_bev_rotation(frame_idx)
            # TODO: test if better to rotate first and then take step
            states_to_plot = [current_rotation @ (s - self.dT * current_velocity) for s in states_to_plot]
            states_to_plot += current_state

        _x = [x[0, 0] for x in states_to_plot]
        _z = [x[1, 0] for x in states_to_plot]
        return _x, _z

    def plot_bev_gt_trajectories(self):
        _x, _z = self.get_bev_gt_trajectories(up_to_frame_idx=self.max_frame_idx)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(5, 5, forward=True)
        plt.plot(_x, _z, 'o', ms=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title('History of ground truth targets')
        ax.grid(True)
        plt.show()

    def plot_ego_bev_motion(self):
        ego_states = self.get_ego_bev_motion()
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(5, 5, forward=True)
        plt.plot(ego_states[0], ego_states[1], 'o', ms=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title('History of ego motion')
        ax.grid(True)
        plt.show()













