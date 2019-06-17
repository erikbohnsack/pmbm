import pickle
import os
from utils.eval_metrics import GOSPA
import numpy as np
import glob
from .mot_metrics import MotCalculator
import motmetrics as mm
import pandas as pd
import time
from utils.coord_transf import within_fov
import math

class Logger:
    def __init__(self, sequence_idx, config_idx, filename=''):
        if not os.path.exists('logs'):
            os.mkdir('logs')

        self.sequence_idx = sequence_idx
        timestr = time.strftime("%Y%m%d-%H%M%S")

        self.filename = 'logs/log-' + filename + '-' + timestr
        if os.path.exists(self.filename):
            os.remove(self.filename)

        self.statsname = 'logs/' + 'stats_cfg_' + str(config_idx).zfill(2) + '_' + filename + '_seq' + str(sequence_idx).zfill(4) + '-' + timestr
        if os.path.exists(self.statsname):
            os.remove(self.statsname)

    def log_data(self, PMBM, frame_id, total_time=-1, measurements=None, true_states=None, verbose=False):
        data = self.encode_data(PMBM, frame_id, total_time, measurements, true_states)

        if verbose:
            self.display_data(data)

        with open(self.filename, 'ab') as fp:
            pickle.dump(data, fp)

    def encode_data(self, PMBM, frame_id, total_time, measurements, true_states):
        data = {
            'state_dims': PMBM.state_dims,
            'current_time': PMBM.current_time,
            'measurements': measurements,
            'true_states': true_states,
            'estimated_targets': PMBM.estimated_targets,
            'global_hypos': PMBM.global_hypotheses,
            'targets': PMBM.targets,
            'nof_global_hypos': len(PMBM.global_hypotheses),
            'nof_targets': len(PMBM.targets),
            'nof_STH': count_nof_STH(PMBM),
            'nof_ET': len(PMBM.estimated_targets),
            'total_time': total_time,
            'frame_id': frame_id
        }
        return data

    def display_data(self, data):
        print('\t#targets: \t {} \n\t#STH: \t {} \n\t#global hypos: \t {} \n\t#est.targets \t {}'.format(data['nof_targets'],
                                                                                                         data['nof_STH'],
                                                                                       data['nof_global_hypos'],
                                                                                       len(data['estimated_targets'])))

    def log_stats(self, total_time_per_iteration, gospa_score_list, mot_summary, config, predictions_average_gospa,
                  verbose=False):
        data = {
            'config_name' : config.name,
            'sequence_idx': self.sequence_idx,
            'time_per_iter': total_time_per_iteration,
            'gospa_sl': gospa_score_list,
            'mot_summary' : mot_summary,
            'motion_model' : config.motion_model_name,
            'poisson_states_model_name' : config.poisson_states_model_name,
            'filter_name' : config.filter_name,
            'predictions_average_gospa' : predictions_average_gospa
        }
        if verbose:
            self.display_data(data)

        with open(self.statsname, 'wb') as fp:
            pickle.dump(data, fp)

def count_nof_STH(PMBM):
    counter = 0
    for target in PMBM.targets.values():
        for single in target.single_target_hypotheses:
            counter += 1
    return counter


def load_logging_data(filename):
    data = []
    with open(filename, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data

def create_estimated_trajectory(data=False, filename=False):
    if filename:
        data = load_logging_data(filename)
    target_indeces = []
    estimated_time = {}
    estimated_states = {}
    estimated_variances = {}
    estimated_predicted_states = {}
    estimated_predicted_variances = {}

    for d in data:
        current_time = d['current_time']
        for est in d['estimated_targets']:
            target_idx = est['target_idx']
            target_state = est['single_target'].state
            target_variance = est['single_target'].variance
            predicted_states = est['state_predictions']
            predicted_variances = est['var_predictions']
            if target_idx in target_indeces:
                estimated_states[target_idx].append(target_state)
                estimated_variances[target_idx].append(target_variance)
                estimated_time[target_idx].append(current_time)
                estimated_predicted_states[target_idx].append(predicted_states)
                estimated_predicted_variances[target_idx].append(predicted_variances)
            else:
                estimated_states[target_idx] = [target_state]
                estimated_variances[target_idx] = [target_variance]
                estimated_time[target_idx]= [current_time]
                estimated_predicted_states[target_idx] = [predicted_states]
                estimated_predicted_variances[target_idx] = [predicted_variances]
                target_indeces.append(target_idx)
    return estimated_states, estimated_variances, estimated_time, estimated_predicted_states, estimated_predicted_variances

def create_hypos_over_time(data=False, filename=False):
    if filename:
        data = load_logging_data(filename)
    time = []
    nof_globals = []
    nof_targets = []
    nof_sths = []
    nof_ets = []
    nof_gts = []
    for d in data:
        time.append(d['current_time'])
        nof_globals.append(d['nof_global_hypos'])
        nof_targets.append(d['nof_targets'])
        nof_sths.append(d['nof_STH'])
        nof_ets.append(d['nof_ET'])
        nof_gts.append(len(d['true_states']))
    return time, nof_globals, nof_targets, nof_sths, nof_ets, nof_gts

def create_total_time_over_iteration(data=False, filename=False):
    if filename:
        data = load_logging_data(filename)
    total_times = []
    iterations = []
    for d in data:
        total_times.append(d['total_time'])
        iterations.append(d['current_time'])

    return iterations, total_times

def data_states_2_gospa(d, gt_dims):
    state_dims = d['state_dims']
    true_states = np.array([], dtype=np.float).reshape(0, gt_dims)
    for ts in d['true_states']:
        true_states = np.vstack((true_states, ts.reshape(1, gt_dims)))

    if gt_dims == state_dims:
        estimated_targets = d['estimated_targets']
        estimated_states =  np.array([], dtype=np.float).reshape(0,4)
        for et in estimated_targets:
            _state = et['single_target'].state
            estimated_states = np.vstack((estimated_states, _state.reshape(1,state_dims)))
    elif gt_dims < state_dims:
        estimated_targets = d['estimated_targets']
        estimated_states = np.array([], dtype=np.float).reshape(0, gt_dims)
        for et in estimated_targets:
            _state = et['single_target'].state[0:gt_dims]
            estimated_states = np.vstack((estimated_states, _state.reshape(1, gt_dims)))
    else:
        assert 'Estimated states not as big as ground truth states, do smth about it and then come back'

    return true_states, estimated_states, state_dims


def calculate_GOSPA_score(data=False, filename=False, gt_dims=4):
    if filename:
        data = load_logging_data(filename)
    total_score = 0
    score_list = []
    for d in data:
        true_states, estimated_states, state_dims = data_states_2_gospa(d, gt_dims)
        score = GOSPA(true_states, estimated_states, p=1, c=100, alpha=2., state_dim=gt_dims)
        total_score += score
        score_list.append(score)
    mean_score = np.mean(score_list)
    print('\n==========================\n'
          'Total GOSPA score: {}\n'
          'Average GOSPA score: {}'
          '\n=========================='.format(round(total_score, 2), round(mean_score, 2)))
    return score_list


def calculate_MOT(sequence_idx, root, data=False, filename=False, classes_to_track=['all']):
    if not data and not filename:
        raise ValueError("Neither data nor filename provided.")

    if filename and not data:
        data = load_logging_data(filename)
    acc = MotCalculator(sequence_idx, path_to_data=root, classes_to_track=classes_to_track)
    #n_unique_id = 0
    for d in data:
        frame_id = d['frame_id']
        estimated_targets = d['estimated_targets']
        #try:
        #    max_track_id = max([lbl.track_id for lbl in acc.kitti.lbls[frame_id]])
        #except:
        #    max_track_id = n_unique_id
        #if max_track_id > n_unique_id:
        #    n_unique_id = max_track_id
        acc.calculate(estimated_targets, frame_id)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'mostly_tracked', 'mostly_lost',
                                       'num_false_positives', 'num_switches', 'num_fragmentations'], name=str(sequence_idx))

    # Make MOTP higher = better
    summary['motp'] = 1 - summary['motp']

    # If weird stuff due to no ground truth for current class_to_track happens, fix it
    if np.isnan(summary['motp'].values[0]):
        summary['motp'] = 1.
    if np.isnan(summary['mota'].values[0]):
        summary['mota'] = 1.
    if np.isposinf(summary['motp'].values[0]) or np.isneginf(summary['motp'].values[0]):
        summary['motp'] = 0.
    if np.isposinf(summary['mota'].values[0]) or np.isneginf(summary['mota'].values[0]):
        summary['mota'] = 0.

    #summary.at[str(sequence_idx), 'mostly_tracked'] /= n_unique_id
    #summary.at[str(sequence_idx), 'mostly_lost'] /= n_unique_id
    #summary.at[str(sequence_idx), 'num_false_positives'] /= n_unique_id
    #summary.at[str(sequence_idx), 'num_switches'] /= n_unique_id
    #summary.at[str(sequence_idx), 'num_fragmentations'] /= n_unique_id
    print(summary)
    return summary


def prediction_stats(sequence_idx, config, kitti, data=False, filename=False):
    if not data and not filename:
        raise ValueError("Neither data nor filename provided.")
    if filename and not data:
        data = load_logging_data(filename)

    imu_dict = kitti.imus

    dt = config.dt
    gt_dims = 2
    print('Sequence: {}'.format(sequence_idx))
    all_true_states = []
    for d in data:
        all_true_states.append(d['true_states'])

    gospa_scores = np.zeros(config.show_predictions)
    for d in data:
        #print('\tCurrent time: {}'.format(d['current_time']))
        if d['current_time'] > len(data) - config.show_predictions:
            #print('*** No more ground truths to work with... Aborting ***')
            break

        future_ego_position = np.array([0, 0], dtype=np.float).reshape(gt_dims, 1)
        track_indeces = []

        for t in range(config.show_predictions):
            state_dims = d['state_dims']
            imu_data = imu_dict[d['current_time'] + t]
            angle = imu_data.ru * dt
            rotation_matrix = np.array([[math.cos(angle), - math.sin(angle)],
                                        [math.sin(angle), math.cos(angle)]])
            translation = np.array([[-imu_data.vl * dt],
                                    [imu_data.vf * dt]])
            future_ego_position = rotation_matrix @ future_ego_position + translation

            true_states = np.array([], dtype=np.float).reshape(0, gt_dims)
            for lbl in kitti.lbls[d['current_time'] + t]:
                if t == 0:
                    track_indeces.append(lbl.track_id)
                if not lbl.track_id in track_indeces:
                    continue

                ts = np.array([[lbl.location[0]], [lbl.location[2]]])
                _ts = ts + future_ego_position
                true_states = np.vstack((true_states, _ts.reshape(1, gt_dims)))

            estimated_states = np.array([], dtype=np.float).reshape(0, gt_dims)
            for et in d['estimated_targets']:
                _state = et['state_predictions'][t][0:gt_dims]

                if within_fov(_state, min_angle=config.uniform_angle[0],
                                      max_angle=config.uniform_angle[1],
                                      max_radius=config.uniform_radius):
                    estimated_states = np.vstack((estimated_states, _state.reshape(1, gt_dims)))

            score = GOSPA(true_states, estimated_states, p=1, c=100, alpha=2., state_dim=gt_dims)
            gospa_scores[t] += score

    mean_gospas = gospa_scores / (len(data) - config.show_predictions)

    return gospa_scores, mean_gospas
    #print('\tTotal GOSPAs: \n\t{}\n\tAverage GOSPAs: \n\t{}'.format(gospa_scores, mean_gospas))


def fafe_prediction_stats(sequence_idx, kitti, data=False, filename=False, num_conseq_frames=5):
    if not data and not filename:
        raise ValueError("Neither data nor filename provided.")
    if filename and not data:
        data = load_logging_data(filename)

    imu_dict = kitti.imus
    dt = 0.1
    gt_dims = 2
    print('Sequence: {}'.format(sequence_idx))
    all_true_states = []
    for d in data:
        all_true_states.append(d['true_states'])

    gospa_scores = np.zeros(num_conseq_frames)
    for d in data:
        future_ego_position = np.array([0, 0], dtype=np.float).reshape(gt_dims, 1)
        track_indeces = []

        for t in range(num_conseq_frames):
            state_dims = d['state_dims']
            imu_data = imu_dict[d['current_time'] + t]
            angle = imu_data.ru * dt
            rotation_matrix = np.array([[math.cos(angle), - math.sin(angle)],
                                        [math.sin(angle), math.cos(angle)]])
            translation = np.array([[-imu_data.vl * dt],
                                    [imu_data.vf * dt]])
            future_ego_position = rotation_matrix @ future_ego_position + translation

            true_states = np.array([], dtype=np.float).reshape(0, gt_dims)
            for lbl in kitti.lbls[d['current_time'] + t]:
                if t == 0:
                    track_indeces.append(lbl.track_id)
                if not lbl.track_id in track_indeces:
                    continue

                ts = np.array([[lbl.location[0]], [lbl.location[2]]])
                _ts = ts + future_ego_position
                true_states = np.vstack((true_states, _ts.reshape(1, gt_dims)))

            estimated_states = np.array([], dtype=np.float).reshape(0, gt_dims)
            for et in d['estimated_targets']:
                if t >= len(et['state_predictions']):
                    continue

                _state = et['state_predictions'][t][0:gt_dims]

                if within_fov(_state, min_angle=0.78,
                                      max_angle=2.35,
                                      max_radius=100):
                    estimated_states = np.vstack((estimated_states, _state.reshape(1, gt_dims)))

            score = GOSPA(true_states, estimated_states, p=1, c=100, alpha=2., state_dim=gt_dims)
            gospa_scores[t] += score

    mean_gospas = gospa_scores / len(data)

    return gospa_scores, mean_gospas


def create_sequences_stats(filenames_prefix='logs/stats_seq'):
    filenames = glob.glob(filenames_prefix + '*')
    sequences = []
    avg_times = []
    avg_gospa = []
    mota = []
    motp = []
    columns = ['SeqId', 'CfgName', 'MotionModel', 'Filter', 'PoissonModel', '#Frames', 'Pase', 'MOTA', 'MOTP', 'GOSPA',
               'MostlyTracked', 'MostlyLost', '#FalsePos', '#Switches', '#Fragmentations', 'PredGOSPA']
    rows = []
    for ix, filename in enumerate(filenames):
        data = load_logging_data(filename)

        d = data[0]
        sequences.append(d['sequence_idx'])
        avg_times.append(np.mean(d['time_per_iter']))
        avg_gospa.append(np.mean(d['gospa_sl']))
        mota.append(float(d['mot_summary']['mota'].values))
        motp.append(float(d['mot_summary']['motp'].values))
        df = d['mot_summary']
        df['gospa'] = np.mean(d['gospa_sl'])
        df['avg_times'] =np.mean(d['time_per_iter'])
        df['motion_model'] = d['motion_model']
        df['poisson_model'] = d['poisson_states_model_name']
        df['filter'] = d['filter_name']

        _seqid = d['sequence_idx']
        _name = d['config_name']
        _motionmodel = d['motion_model']
        _filter = d['filter_name']
        _poissonmodel = d['poisson_states_model_name']
        _nframes = int(df['num_frames'].values)
        _pase = round(float(np.mean(d['time_per_iter'])), 3)
        _mota = round(float(d['mot_summary']['mota'].values), 2)
        _motp = round(float(d['mot_summary']['motp'].values), 2)
        _gospa = round(float(np.mean(d['gospa_sl'])), 2)
        _pred_gospa = d['predictions_average_gospa']
        _mostly_tracked = round(float(df['mostly_tracked'].values), 2)
        _mostly_lost = round(float(df['mostly_lost'].values), 2)
        _nfalsepos = round(float(df['num_false_positives'].values), 2)
        _nswitches = round(float(df['num_switches'].values), 2)
        _nfrags = round(float(df['num_fragmentations'].values), 2)

        row = [_seqid, _name, _motionmodel, _filter, _poissonmodel, _nframes, _pase, _mota, _motp, _gospa, _mostly_tracked,
               _mostly_lost, _nfalsepos, _nswitches, _nfrags, _pred_gospa]

        rows.append(row)
    stats_df = pd.DataFrame(np.array(rows),
                            columns=columns)

    return sequences, avg_times, avg_gospa, mota, motp, stats_df

def create_sequence_dataframe(filenames_prefix='logs/stats_seq'):
    filenames = glob.glob(filenames_prefix + '*')
    columns = ['SeqId', 'CfgName', 'MotionModel', 'Filter', 'PoissonModel', '#Frames', 'Pase', 'MOTA', 'MOTP', 'GOSPA',
               'MostlyTracked', 'MostlyLost', '#FalsePos', '#Switches', '#Fragmentations', 'PredGOSPA']
    rows = []
    for ix, filename in enumerate(filenames):
        data = load_logging_data(filename)
        d = data[0]
        df = d['mot_summary']
        _seqid = d['sequence_idx']
        _name = d['config_name']
        _motionmodel = d['motion_model']
        _filter = d['filter_name']
        _poissonmodel = d['poisson_states_model_name']
        _nframes = int(df['num_frames'].values)
        _pase = round(float(np.mean(d['time_per_iter'])), 3)
        _mota = round(float(d['mot_summary']['mota'].values), 2)
        _motp = round(float(d['mot_summary']['motp'].values), 2)
        _gospa = round(float(np.mean(d['gospa_sl'])), 2)
        _pred_gospa = d['predictions_average_gospa']
        _mostly_tracked = round(float(df['mostly_tracked'].values), 2)
        _mostly_lost = round(float(df['mostly_lost'].values), 2)
        _nfalsepos = round(float(df['num_false_positives'].values), 2)
        _nswitches = round(float(df['num_switches'].values), 2)
        _nfrags = round(float(df['num_fragmentations'].values), 2)
        row = [_seqid, _name, _motionmodel, _filter, _poissonmodel, _nframes, _pase, _mota, _motp, _gospa, _mostly_tracked,
               _mostly_lost, _nfalsepos, _nswitches, _nfrags, _pred_gospa]
        rows.append(row)
    stats_df = pd.DataFrame(np.array(rows),
                            columns=columns)
    return stats_df


def measurements_from_log(data, frame_idx):
    return data[frame_idx]['measurements']




