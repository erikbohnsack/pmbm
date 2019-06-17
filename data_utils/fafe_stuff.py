import numpy as np
import os
from utils import logger
from scipy.spatial.distance import cdist

class FafeDetections:
    def __init__(self, showroom_path, sequence):
        file_path = os.path.join(showroom_path, 'detections_' + str(sequence).zfill(4) + '.txt')
        assert os.path.exists(file_path), 'File does not exist: {}'.format(file_path)

        with open(file_path, 'r') as f:
            lines = f.read().splitlines()

        detections = {}
        frames = []
        max_frame = 0

        for line in lines:
            l = line.split(' ')
            _frame = int(l[0])
            _class = l[1]
            _x = float(l[2])
            _y = float(l[3])
            _r = float(l[4])
            if _frame not in frames:
                frames.append(_frame)
                detections[_frame] = [[_x, _y, _r, _class]]
            else:
                detections[_frame].append([_x, _y, _r, _class])

            if _frame > max_frame:
                max_frame =  _frame

        # Add empty measurements in frames where we didn't find anyting
        for f in np.arange(0, max_frame + 1):
            if f not in frames:
                detections[f] = []

        self.detections = detections
        self.max_frame_idx = max_frame

    def get_fafe_detections(self, frame_idx, measurement_dims):
        measurements = []
        classes = []
        for meas in self.detections[frame_idx]:
            measurements.append(np.array(meas[0:measurement_dims]).reshape(measurement_dims,1))
            classes.append(meas[-1])
        return measurements, classes

class FafeMeasurements:
    def __init__(self, log_path):

        data = logger.load_logging_data(log_path)
        self.frames = []
        self.measurements = []

        for d in data[0]:
            self.frames.append(d['frame_id'])
            self.measurements.append(remove_duplicates(d['measurements']))

        self.num_skip = self.frames[0]

    def get_fafe_measurements(self, frame_idx, measurement_dims):
        if frame_idx in self.frames:
            out_meas = []
            out_cls  = []
            for meas in self.measurements[frame_idx-self.num_skip]:
                out_meas.append(np.array(meas[0:measurement_dims]).reshape(measurement_dims,1))
                out_cls.append('Car')
            return out_meas, out_cls
        else:
            return [], []

def remove_duplicates(measurements):
    meas = []
    for ir in measurements:
        a = ir[0][0]
        b = ir[1][0]
        c = [a, b]
        if c not in meas:
            meas.append(c)
    return np.array(meas)


