import motmetrics as mm
import numpy as np
from data_utils import kitti_stuff


class MotCalculator(mm.MOTAccumulator):
    def __init__(self, sequence_idx, auto_id=False, path_to_data='/Users/erikbohnsack/data', split='training',
                 classes_to_track=['all']):
        super().__init__(auto_id=auto_id)

        self.kitti = kitti_stuff.Kitti(ROOT=path_to_data, split=split)
        self.kitti.lbls = self.kitti.load_labels(sequence_idx)
        self.classes_to_track = classes_to_track

    def calculate(self, estimated_targets, frameid):
        """
        Calculates
        :param estimated_targets: list of dicts. [{'single_target':, 'single_hypo_idx':, 'target_idx':},...]
        :param frameid:
        :return:
        """

        # Rearrange ground truths to be able to input to PyMotMetrics
        ground_truth = self.kitti.lbls[frameid]
        gt_ids = []
        o = np.array([], dtype=np.float).reshape(0, 2)
        h = np.array([], dtype=np.float).reshape(0, 2)
        for gt_label in ground_truth:
            if gt_label.type[0] == 'DontCare':
                continue
            if 'all' not in self.classes_to_track and gt_label.type[0] not in self.classes_to_track:
                continue

            gt_ids.append(gt_label.track_id)

            o = np.vstack((o, np.array([[gt_label.location[0], gt_label.location[2]]])))

        # Rearrange estimated targets to be able to input to PyMotMetrics
        est_ids = []
        for target in estimated_targets:
            est_ids.append(target['target_idx'])
            state = target['single_target'].state.T
            if state.shape[1] > 2:
                state = state[:, 0:2]
            assert state.shape == (1, 2), 'State not row vector. '
            h = np.vstack((h, state))

        # Calculate dists
        dists = mm.distances.norm2squared_matrix(o, h, max_d2=5.)

        self.update(oids=gt_ids, hids=est_ids, dists=dists, frameid=frameid)

        return


    #mh = mm.metrics.create()

    #summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')

