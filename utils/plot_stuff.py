import numpy as np
from matplotlib.patches import Ellipse
from .logger import load_logging_data, create_estimated_trajectory, create_hypos_over_time
from .logger import measurements_from_log, create_total_time_over_iteration, create_sequences_stats
from .logger import create_sequence_dataframe
import matplotlib.pyplot as plt
from data_utils.kitti_stuff import Kitti
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils.constants
from matplotlib import gridspec
import matplotlib as mpl
import os, datetime
import pandas as pd
import cv2
import glob

XLIM = utils.constants.XLIM
ZLIM = utils.constants.ZLIM


def plot_estimated_targets(data=None, filename=None, measurements=None, true_trajectories=None, save=False):
    estimated_states, estimated_variances, estimated_time = create_estimated_trajectory(data, filename)
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(20, 20 / 3, forward=True)

    for key, est in estimated_states.items():
        _x = [e[0] for e in est]
        _y = [e[1] for e in est]
        _vx = [e[2] for e in est]
        _vy = [e[3] for e in est]
        _t = estimated_time[key]

        ax[0].plot(_x, _y, 'o-', lw=2, ms=2)
        ax[1].plot(_t, _vx, 'o-', lw=2, ms=2)
        ax[2].plot(_t, _vy, 'o-', lw=2, ms=2)

    if not true_trajectories is None:
        for tt in true_trajectories:
            t_x = [x[0] for x in tt]
            t_y = [x[1] for x in tt]
            ax[0].plot(t_x, t_y, 'ko-', lw=0.5, ms=2)
            v_x = [x[2] for x in tt]
            v_y = [x[3] for x in tt]
            t = np.arange(0, len(v_x), 1)
            ax[1].plot(t, v_x, 'ko-', lw=0.5, ms=2)
            ax[2].plot(t, v_y, 'ko-', lw=0.5, ms=2)

    if not measurements is None:
        for current in measurements:
            for meas in current:
                try:
                    m_x = meas[0]
                    m_y = meas[1]
                except:
                    continue
                ax[0].plot(m_x, m_y, 'k+', ms=4, mfc='none')

    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Position')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('vx')
    ax[1].set_title('x-Velocity')
    ax[2].set_xlabel('t')
    ax[2].set_ylabel('vy')
    ax[2].set_title('y-Velocity')


def plot_hypos_over_time(data=None, filename=None, save=False, config_name=None):
    time, nof_globals, nof_targets, nof_sths, nof_ets, nof_gts = create_hypos_over_time(data, filename)
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(20, 20 / 4, forward=True)
    ax[0].plot(time, nof_globals, 'o-', lw=1, ms=2)
    ax[1].plot(time, nof_targets, 'o-', lw=1, ms=2)
    ax[2].plot(time, nof_sths, 'o-', lw=1, ms=2)
    ax[3].plot(time, nof_ets, 'bo-', lw=1, ms=2, label='et')
    ax[3].plot(time, nof_gts, 'ro-', lw=1, ms=2, label='gt')

    ax[0].set_xlabel('Time step')
    ax[0].set_ylabel('#')
    ax[0].set_title('#Global hypotheses')
    ax[1].set_xlabel('Time step')
    ax[1].set_ylabel('#')
    ax[1].set_title('#Tracks')
    ax[2].set_xlabel('Time step')
    ax[2].set_ylabel('#')
    ax[2].set_title('#STHs')
    ax[3].set_xlabel('Time step')
    ax[3].set_ylabel('#')
    ax[3].set_title('#Estimated targets')
    ax[3].legend()

    if config_name is not None:
        if not os.path.exists('showroom'):
            os.mkdir('showroom')
        fig.savefig('showroom/hypos_over_time_' + config_name + '.pdf')
    plt.show(fig)


def plot_total_time_per_iteration(data=None, filename=None, save=False, config_name=None):
    iterations, total_times = create_total_time_over_iteration(data, filename)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5, 5, forward=True)
    ax.plot(iterations, total_times, 'o-', lw=1, ms=2)
    ax.set_xlabel('iteration k')
    ax.set_ylabel('time [s]')
    ax.set_title('Time to run one iteration')

    if config_name is not None:
        if not os.path.exists('showroom'):
            os.mkdir('showroom')
        fig.savefig('showroom/total_time_' + config_name + '.pdf')
    plt.show(fig)


def plot_time_score(data=None, filename=None, save=False, score_list=None, config_name=None):
    iterations, total_times = create_total_time_over_iteration(data, filename)
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 5, forward=True)
    ax[0].plot(iterations, total_times, 'o-', lw=1, ms=2)
    ax[0].set_xlabel('Time step')
    ax[0].set_ylabel('time [s]')
    ax[0].set_title('Time to run one iteration')

    ax[1].plot(iterations, score_list, 'o-', lw=1, ms=2)
    ax[1].set_xlabel('Time step')
    ax[1].set_ylabel('score [-]')
    ax[1].set_title('GOSPA Score')

    if config_name is not None:
        if not os.path.exists('showroom'):
            os.mkdir('showroom')
        fig.savefig('showroom/time_score_' + config_name + '.pdf')
    plt.show(fig)


def plot_target_life(data=None, filename=None, save=False, config_name=None):
    estimated_states, estimated_variances, estimated_time, _, _ = create_estimated_trajectory(data, filename)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5, 15, forward=True)
    for key, et_time in estimated_time.items():
        key_vector = np.ones((len(et_time, ))) * key
        ax.plot(et_time, key_vector, '-', lw=2)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Track ID')
        ax.set_title('Life span of each track')

    if config_name is not None:
        if not os.path.exists('showroom'):
            os.mkdir('showroom')
        fig.savefig('showroom/target_life_' + config_name + '.pdf')
    plt.show(fig)


def plot_tracking_history(path, sequence_idx, data=False, filename_log=False, kitti=Kitti,
                          final_frame_idx=None, disp='show', only_alive=False, show_cov=False, show_predictions=None,
                          config_name='', car_van_flag=False, fafe=False, num_conseq_frames=None):

    if fafe and num_conseq_frames is None:
        raise ValueError("Fafe needs num conseq frames")
    if not data and not filename_log:
        raise ValueError("Neither data or filename to log file specified. ")

    if filename_log and not data:
        data = load_logging_data(filename_log)

    if not os.path.exists(path):
        os.mkdir(path)

    fig, ax = plt.subplots(2, 1)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
    fig.set_size_inches(10, 15, forward=True)

    img = kitti.load_image(sequence_idx=sequence_idx, frame_idx=final_frame_idx)
    ax[0].imshow(img)
    ax[0].grid(False)

    ego_vehicle = patches.Rectangle((-0.5, -2), 1, 4, color="blue", alpha=0.50)
    ax[1].add_patch(ego_vehicle)

    for l in kitti.lbls[final_frame_idx]:

        if car_van_flag:
            if l.type[0] not in ['Car', 'Van']:
                continue
        else:
            if l.type[0] == 'DontCare':
                continue

        x_pos = l.location[0]
        z_pos = l.location[2]
        pos = np.array([x_pos, z_pos])
        width = l.dimensions[1]
        length = l.dimensions[2]

        if x_pos <= XLIM[0] or x_pos >= XLIM[1] or z_pos <= ZLIM[0] or z_pos >= ZLIM[1]:
            continue

        rot_y = l.rotation_y
        _xm = - width / 2
        _zm = - length / 2
        _xp = width / 2
        _zp = length / 2

        _bbox = np.matrix([[_xm, _zm], [_xm, _zp], [_xp, _zp], [_xp, _zm]])
        _phi = np.pi / 2 - rot_y
        _rotm = np.matrix([[np.cos(_phi), -np.sin(_phi)], [np.sin(_phi), np.cos(_phi)]])
        _rotated_bbox = (_rotm * _bbox.T).T + pos
        r = patches.Polygon(_rotated_bbox, color="red", alpha=0.2)
        ax[1].add_patch(r)
        ax[1].text(x_pos + 0.5 * width, z_pos + 0.5 * length, str(l.track_id), color='black')
        ax[1].plot(x_pos, z_pos, 'r.', ms=0.5)

    # Plot current measurements. If fafe the measurements is lined up differently.
    if fafe:
        meas_frame_idx = final_frame_idx - num_conseq_frames + 1
    else:
        meas_frame_idx = final_frame_idx
    measurements = measurements_from_log(data=data, frame_idx=meas_frame_idx)
    for meas in measurements:
        ax[1].plot(meas[0], meas[1], 'rs', markerfacecolor='none')

    es, ev, et, eps, epv = create_estimated_trajectory(data=data)

    for tid, state in es.items():
        frame_indeces = et[tid]
        if only_alive:
            if final_frame_idx not in frame_indeces:
                continue
        last_idx_to_plot = frame_indeces.index(final_frame_idx)
        states_to_plot = []
        for idx, frame_idx in enumerate(frame_indeces[0:last_idx_to_plot + 1]):
            current_state = state[idx][0:2]
            for i in range(frame_idx, final_frame_idx):
                current_velocity = kitti.get_ego_bev_velocity(frame_idx=i)
                current_rotation = kitti.get_ext_bev_rotation(frame_idx=i)
                current_state = current_rotation @ (current_state - kitti.dT * current_velocity)
            states_to_plot.append(current_state)
        _ex = [x[0, 0] for x in states_to_plot]
        _ez = [x[1, 0] for x in states_to_plot]
        _c = cnames[tid % len(cnames)]
        ax[1].plot(_ex, _ez, color=_c, linewidth=2)
        ax[1].plot(_ex[last_idx_to_plot], _ez[last_idx_to_plot], color=_c, marker='o', markerfacecolor='none', ms=5)
        if not (_ex[-1] <= XLIM[0] or _ex[-1] >= XLIM[1] or _ez[-1] <= ZLIM[0] or _ez[-1] >= ZLIM[1]):
            ax[1].text(_ex[last_idx_to_plot] - 1, _ez[last_idx_to_plot] + 1, str(tid), color=_c)

        if show_cov:
            _cov = ev[tid][last_idx_to_plot][0:2, 0:2]
            _cnt = np.array([[_ex[last_idx_to_plot]], [_ez[last_idx_to_plot]]])
            _ = plot_cov_ellipse(_cov, _cnt, nstd=3, ax=ax[1], alpha=0.5, color=_c)

        if show_predictions is not None:
            _pex = [x[0, 0] for x in eps[tid][last_idx_to_plot]]
            _pez = [x[1, 0] for x in eps[tid][last_idx_to_plot]]
            ax[1].plot(_pex, _pez, linestyle='--', marker='^', color=_c, linewidth=0.5, ms=4)

    ax[1].set_xlim(XLIM[0], XLIM[1])
    ax[1].set_ylim(ZLIM[0], ZLIM[1])
    ax[1].grid(True)
    plt.tight_layout()

    if disp == 'show':
        plt.show()
    elif disp == 'save':
        fig.savefig(
            path + '/' + config_name + '_track_seq_' + str(sequence_idx).zfill(4) + '_frame_' + str(final_frame_idx).zfill(
                4) + '.png')
        plt.close(fig)
    else:
        assert 'Noob'

    # return es, ev, et, eps, epv


def plot_sequence_stats(filenames_prefix='logs/stats', disp='show'):
    sequences, avg_times, avg_gospa, mota, motp, df = create_sequences_stats(filenames_prefix=filenames_prefix)
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(20, 5, forward=True)
    ax[0].bar(sequences, avg_times)
    ax[1].bar(sequences, avg_gospa)
    ax[2].bar(sequences, mota)
    ax[3].bar(sequences, motp)

    ax[0].set_xlabel('Sequence [idx]')
    ax[0].set_ylabel('Time [s]')
    ax[0].set_title('Average time per time step')
    ax[1].set_xlabel('Sequence [idx]')
    ax[1].set_ylabel('Mean GOSPA score')
    ax[1].set_title('Mean GOSPA')
    ax[2].set_xlabel('Sequence [idx]')
    ax[2].set_ylabel('MOTA score')
    ax[2].set_title('MOTA')
    ax[3].set_xlabel('Sequence [idx]')
    ax[3].set_ylabel('MOTP score')
    ax[3].set_title('MOTP')

    # For pred gospas (prego)
    fig_prego, ax_prego = plt.subplots(1, 1)
    fig_prego.set_size_inches(10, 10, forward=True)
    ax_prego.set_xlabel('Timesteps ahead')
    ax_prego.set_ylabel('Average GOSPA')
    time_steps_ahead = np.array(range(1, len(df.PredGOSPA[0])+1))
    for a in df.PredGOSPA:
        ax_prego.plot(time_steps_ahead, a, '-o')

    if disp == 'show':
        plt.show()
    elif disp == 'save':
        if not os.path.exists('showroom'):
            os.mkdir('showroom')
        fig.savefig('showroom/sequence_stats' + '.pdf')
        fig_prego.savefig('showroom/prego_stats' + '.pdf')
        plt.close(fig)
        plt.close(fig_prego)
    elif disp == 'all':
        if not os.path.exists('showroom'):
            os.mkdir('showroom')
        fig.savefig('showroom/sequence_stats' + '.pdf')
        fig_prego.savefig('showroom/prego_stats' + '.pdf')
        plt.show()
    else:
        assert 'Noob'


    return df


def sequence_analysis(filenames_prefix='logs/stats', sortby='MotionModel'):
    df = create_sequence_dataframe(filenames_prefix)

    unique_values = np.unique(df.filter(items=[sortby]).values)
    n_unique_sequences = len(np.unique(df.filter(items=['SeqId']).values))

    if len(unique_values) < 2:
        print('Found 1 unique setting. Plotting sequence stats...')
        return plot_sequence_stats(filenames_prefix=filenames_prefix, disp='save'), None

    nof_uv = len(unique_values)
    print('Found {} unique settings and {} unique sequences. Plotting sequence stats...'.format(nof_uv, n_unique_sequences))

    row_names = ['AvgPase', 'AvgMOTA', 'AvgMOTP', 'AvgGOSPA', 'AvgMT', 'AvgSwitches', 'AvgML']

    #fig, ax = plt.subplots(7, nof_uv, sharey='row')
    fig, ax = plt.subplots(6, nof_uv, sharey='row')
    inch_per_ax = 2
    fig.set_size_inches(int(inch_per_ax * nof_uv), inch_per_ax * len(row_names), forward=True)

    # For pred gospas (prego)
    fig_prego, ax_prego = plt.subplots(1, 1)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    fig_prego.set_size_inches(10, 10, forward=True)
    #time_steps_ahead = np.array(range(1, len(df.PredGOSPA[0]) + 1))

    # For pred gospas (prego)
    fig_sub_prego, ax_sub_prego = plt.subplots(n_unique_sequences, 1)
    fig_sub_prego.set_size_inches(7, n_unique_sequences*7, forward=True)

    rows = []
    max_avg_id_switches = 0
    for i, uv in enumerate(unique_values):
        row = []
        a = df.loc[df[sortby] == uv]
        a = a.sort_values('SeqId')

        seqIds = [int(b) for b in a.SeqId.values]
        pases = [float(b) for b in a.Pase.values]
        motas = [float(b) for b in a.MOTA.values]
        motps = [float(b) for b in a.MOTP.values]
        gospas = [float(b) for b in a.GOSPA.values]
        predicted_gospas = a.PredGOSPA.values
        mts = [float(b) for b in a.MostlyTracked.values]
        mls = [float(b) for b in a.MostlyLost.values]
        switches = [float(b) for b in a['#Switches'].values]

        time_steps_ahead = np.array(range(1, len(predicted_gospas[0]) + 1))

        ax[0, i].bar(seqIds, pases)
        _mean = np.mean(pases)
        ax[0, i].axhline(y=_mean, color='r')
        row.append(_mean)

        ax[1, i].bar(seqIds, motas)
        _mean = np.mean(motas)
        ax[1, i].axhline(y=_mean, color='r')
        row.append(_mean)

        _mean = np.mean(motps)
        ax[2, i].bar(seqIds, motps)
        ax[2, i].axhline(y=_mean, color='r')
        row.append(_mean)

        _mean = np.mean(gospas)
        ax[3, i].bar(seqIds, gospas)
        ax[3, i].axhline(y=_mean, color='r')
        row.append(_mean)

        _mean = np.mean(mts)
        ax[4, i].bar(seqIds, mts)
        ax[4, i].axhline(y=_mean, color='r')
        row.append(_mean)

        _mean = np.mean(switches)
        ax[5, i].bar(seqIds, switches)
        ax[5, i].axhline(y=_mean, color='r')
        row.append(_mean)
        if _mean > max_avg_id_switches:
            max_avg_id_switches = _mean

        _mean = np.mean(mls)
        #ax[5, i].bar(seqIds, mls)
        #ax[5, i].axhline(y=_mean, color='r')
        row.append(_mean)

        rows.append(row)

        ax[0, i].set_xlabel('Sequence [idx]')
        ax[0, i].set_title(uv + '\nAvg time/iter')

        ax[1, i].set_xlabel('Sequence [idx]')
        ax[1, i].set_title(uv + '\nMOTA')

        ax[2, i].set_xlabel('Sequence [idx]')
        ax[2, i].set_title(uv + '\nMOTP')

        ax[3, i].set_xlabel('Sequence [idx]')
        ax[3, i].set_title(uv + '\nMean GOSPA')

        ax[4, i].set_xlabel('Sequence [idx]')
        ax[4, i].set_title(uv + '\nMostly Tracked')

        ax[5, i].set_xlabel('Sequence [idx]')
        ax[5, i].set_title(uv + '\nID Switches')

        #ax[6, i].set_xlabel('Sequence [idx]')
        #ax[6, i].set_title(uv + '\nMostly Lost')

        if uv == 'CV':
            style = '-o'
        elif uv == 'Mixed':
            style = ':s'
        elif uv == 'BC':
            style = '--*'
        elif uv == 'CA' :
            style = '-.+'
        elif uv == 'Car-CV':
            style = '-o'
        elif uv == 'Ped-CV':
            style = ':s'
        elif uv == 'Car-BC':
            style = '--*'
        elif uv == 'Ped-BC':
            style = '-.+'
        else:
            style = '-x'
        mean_pg = np.zeros(len(predicted_gospas[0]) , dtype=float)
        for ix, pg in enumerate(predicted_gospas):
            if n_unique_sequences != 1:
                ax_sub_prego[ix].plot(time_steps_ahead, pg, style,  label=uv + '-seq' + str(seqIds[ix]))
                ax_sub_prego[ix].set_xlabel('Time steps ahead')
                ax_sub_prego[ix].set_ylabel('Average prediction GOSPA')
            else:
                ax_sub_prego.plot(time_steps_ahead, pg, style, label=uv + '-seq' + str(seqIds[ix]))
                ax_sub_prego.set_xlabel('Time steps ahead')
                ax_sub_prego.set_ylabel('Average prediction GOSPA')
            mean_pg = np.add(mean_pg, pg)
        mean_pg = mean_pg / n_unique_sequences
        ax_prego.plot(time_steps_ahead, mean_pg, style, label=uv)

    ax[0, 0].set_ylabel('Time [s]')
    ax[1, 0].set_ylabel('MOTA score')
    ax[2, 0].set_ylabel('MOTP score')
    ax[3, 0].set_ylabel('Mean GOSPA score')
    ax[4, 0].set_ylabel('Mostly Tracked')
    ax[5, 0].set_ylabel('nof ID Switches')
    #ax[6, 0].set_ylabel('Mostly Lost')

    ax[1, 0].set_ylim([0, 1])
    ax[2, 0].set_ylim([0, 1])

    ax[5, 0].set_ylim([0, max_avg_id_switches*1.2])

    ax_prego.set_xlabel('Time steps ahead', fontsize=30)
    ax_prego.set_ylabel('Average GOSPA score', fontsize=30)
    ax_prego.set_title('Average Prediction GOSPA over sequences: \n {}'.format(seqIds))
    ax_prego.legend(fontsize=30)

    if n_unique_sequences != 1:
        for ix in range(n_unique_sequences):
            ax_sub_prego[ix].legend()
            ax_sub_prego[ix].set_title('Sequence ' + str(seqIds[ix]))
    else:
        ax_sub_prego.legend()
        ax_sub_prego.set_title('Sequence ' + str(seqIds[ix]))

    fig.tight_layout()
    fig_prego.tight_layout()
    fig_sub_prego.tight_layout()

    if not os.path.exists('showroom'):
        os.mkdir('showroom')
    fig.savefig('showroom/sequence_stats' + '.pdf')
    fig_prego.savefig('showroom/prego_stats' + '.pdf')
    fig_sub_prego.savefig('showroom/prego_sub_stats' + '.pdf')

    #plt.show()

    I = pd.Index(row_names, name="Metric")
    C = pd.Index(unique_values, name=sortby)

    avg_df = pd.DataFrame(data=rows, index=C, columns=I)

    # PLOT AVERAGE STUFF for pase, mota, motp and gospa
    to_plot = ['AvgPase', 'AvgMOTA', 'AvgMOTP', 'AvgGOSPA']
    titles = ['Average Time per iteration', 'Average MOTA', 'Average MOTP', 'Average GOSPA']
    for mi, metric in enumerate(to_plot):
        avg_fig, avg_ax = plt.subplots(1, 1, sharey='row')
        avg_fig.set_size_inches(10, 10, forward=True)
        styles = []
        names = []
        values = []
        for name in avg_df[metric].keys():
            # For comparing motion models
            if name == 'BC':
                style = '#1f77b4'
            elif name == 'CA':
                style = '#ff7f0e'
            elif name == 'CV':
                style = '#2ca02c'
            elif name == 'Mixed':
                style = '#d62728'
            # For car vs pedestrian tracking
            elif name == 'Car-BC':
                style = '#1f77b4'
            elif name == 'Car-CV':
                style = '#ff7f0e'
            elif name == 'Ped-BC':
                style = '#2ca02c'
            elif name == 'Ped-CV':
                style = '#d62728'
            # For FAFE variants
            elif name == 'bev_NN':
                style = '#1f77b4'
            elif name == 'bev_nn':
                style = '#ff7f0e'
            elif name == 'pp_NN':
                style = '#2ca02c'
            elif name == 'pp_nn':
                style = '#d62728'
                # For fafe vs pmbm
            elif name == 'PMBM-CV-4':
                style = '#1f77b4'
            else:
                style = 'k'
            styles.append(style)
            names.append(name)
            values.append(avg_df[metric][name])
        avg_ax.bar(names, values, color=styles, alpha=1)
        for i, v in enumerate(values):
            avg_ax.text(names[i], v, str(round(v,3)), horizontalalignment='center', fontsize=40)

        avg_ax.set_title(titles[mi], fontsize=40)
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)
        avg_fig.tight_layout()
        avg_fig.savefig('showroom/average_' + metric + '.pdf')
        #plt.show()
        avg_fig.clear()

    return df, avg_df


def compare_pmbm_fafe_gospas(pmbm_sl, fafe_filename, pmbm_pred_gospa):

    pmbm_iterations = np.arange(0, len(pmbm_sl))

    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(10, 10, forward=True)

    with open(fafe_filename, 'r') as f:
        lines = f.read().splitlines()

    gospa_scores = []

    max_frame = 0
    for line in lines:
        l = line.split(' ')
        row = []
        for i in l[1:]:
            row.append(float(i))
        gospa_scores.append(row)

        if int(l[0]) > max_frame:
            max_frame = int(l[0])

    min_frame = int(lines[0].split(' ')[0])
    gospa_scores = np.array(gospa_scores)

    fafe_iterations = np.arange(min_frame, max_frame+1)
    # Prediction average gospa...
    fafe_pred_gospa = np.mean(gospa_scores, axis=0)

    fafe_sl = gospa_scores[:,0]

    ax[0].plot(pmbm_iterations, pmbm_sl, 'o-', lw=1, ms=2, label='PMBM')
    ax[0].plot(fafe_iterations, fafe_sl, 's-', lw=1, ms=2, label='FaFe')

    ax[0].set_xlabel('iteration k')
    ax[0].set_ylabel('score [-]')
    ax[0].set_title('GOSPA Score')
    ax[0].legend()

    pmbm_time_steps_ahead = np.array(range(1, len(pmbm_pred_gospa) + 1))
    fafe_timesteps_ahead = np.array(range(1, min_frame + 2))

    ax[1].plot(pmbm_time_steps_ahead, pmbm_pred_gospa, 'o-', lw=2, ms=3, label='PMBM')
    ax[1].plot(fafe_timesteps_ahead, fafe_pred_gospa, 's-', lw=2, ms=3, label='FaFe')
    ax[1].set_xlabel('Timesteps ahead')
    ax[1].set_ylabel('score [-]')
    ax[1].set_title('GOSPA Prediction Score')
    ax[1].legend()

    fig.tight_layout()
    fig.savefig('showroom/gospa_comparison' + '.pdf')
    plt.show()

def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    USAGE:
        fig, ax = plt.subplots()
        e = get_cov_ellipse(.)
        ax.add_artist(e)
        plt.show()
    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)


def make_movie_from_images(path):
    video_name = "".join((path, ".mov"))
    images = [img for img in os.listdir(path) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))

    cv2.destroyAllWindows()
    video.release()


def plot_bernoulli(local_bernoulli=None, measurements=None):
    fig, ax = plt.subplots()

    # Plot Prior state
    ax.plot(local_bernoulli.state[0], local_bernoulli.state[1], 'ko', label='Prior State')

    # Plot all measurements
    for meas in measurements:
        ax.plot(meas[0], meas[1], 'rs', label='meas')

    # Plot output states as long as they are not None
    # (if None they have been gated away in Bernoulli Class)
    for ix, st in enumerate(local_bernoulli.output_states):
        if st is None:
            continue
        ax.plot(st[0], st[1], 'g.', label='posterior', linewidth=2)

        _e = get_cov_ellipse(local_bernoulli.output_variance[ix], (st[0], st[1]), 3,
                             alpha=local_bernoulli.output_weights[ix])
        ax.add_artist(_e)

    # TODO: make this plotting great again
    ax.set(xlabel='x', ylabel='y', title='hejhopp')

    ax.grid()
    # ax.set_xlim()
    # ax.set_ylim()
    plt.show()


def plot_poissons(poisson):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xlim = 15
    ylim = 30

    for dist in poisson.distributions:
        plot_cov_ellipse(dist.variance[0:2, 0:2], dist.state[0:2], nstd=2, ax=ax)

    plt.xlim(-xlim, xlim)
    plt.ylim(-0.25 * ylim, 1.75 * ylim)
    plt.grid(True)

    plt.show()


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, alpha=1, color='black', **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, alpha=alpha, color=color)

    ax.add_artist(ellip)
    return ellip


cnames = ['black'
    , 'blue'
    , 'blueviolet'
    , 'brown'
    , 'cadetblue'
    , 'chartreuse'
    , 'chocolate'
    , 'coral'
    , 'cornflowerblue'
    , 'crimson'
    , 'darkblue'
    , 'darkcyan'
    , 'darkgoldenrod'
    , 'darkgray'
    , 'darkgreen'
    , 'darkkhaki'
    , 'darkmagenta'
    , 'darkolivegreen'
    , 'darkorange'
    , 'darkorchid'
    , 'darkred'
    , 'darksalmon'
    , 'darkseagreen'
    , 'darkslateblue'
    , 'darkslategray'
    , 'darkturquoise'
    , 'darkviolet'
    , 'deeppink'
    , 'deepskyblue'
    , 'dimgray'
    , 'dodgerblue'
    , 'firebrick'
    , 'forestgreen'
    , 'fuchsia'
    , 'gainsboro'
    , 'gold'
    , 'goldenrod'
    , 'gray'
    , 'green'
    , 'greenyellow'
    , 'hotpink'
    , 'indianred'
    , 'indigo'
    , 'lawngreen'
    , 'lightgreen'
    , 'lime'
    , 'limegreen'
    , 'magenta'
    , 'maroon'
    , 'mediumaquamarine'
    , 'mediumblue'
    , 'mediumorchid'
    , 'mediumpurple'
    , 'mediumseagreen'
    , 'mediumslateblue'
    , 'mediumspringgreen'
    , 'mediumturquoise'
    , 'mediumvioletred'
    , 'midnightblue'
    , 'navy'
    , 'olive'
    , 'olivedrab'
    , 'orange'
    , 'orangered'
    , 'orchid'
    , 'palegoldenrod'
    , 'palegreen'
    , 'palevioletred'
    , 'peru'
    , 'purple'
    , 'red'
    , 'rosybrown'
    , 'royalblue'
    , 'saddlebrown'
    , 'salmon'
    , 'sandybrown'
    , 'seagreen'
    , 'sienna'
    , 'slateblue'
    , 'slategray'
    , 'snow'
    , 'springgreen'
    , 'steelblue'
    , 'teal'
    , 'tomato'
    , 'turquoise'
    , 'violet'
    , 'yellowgreen']
