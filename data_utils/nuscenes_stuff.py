import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from .dataset_stuff import Label, IMU
from utils.constants import PossibleClasses
from nuscenes.nuscenes import NuScenes

class Nuscenes:
    def __init__(self, ROOT):
        self.path = ROOT

        self.label_path = os.path.join(self.path, 'label_2')
        self.image_path = os.path.join(self.path, 'CAM_FRONT')
        self.velo_path = os.path.join(self.path, 'LIDAR_TOP')
        self.imu_path = os.path.join(self.path, 'oxts')

        self.imgs, self.pcds, self.lbls, self.imus = None, None, None, None
        self.max_frame_idx = None

        self.dT = 0.1 # 10 Hz sampling rate for kitti tracking dataset

    def __repr__(self):
        return '<Kitti | Path: {}>'.format(self.path)

