import numpy as np
from pmbm.pmbm import PMBM
from pmbm.config import Config
import time


def test_pmbm(logger, true_trajectories, measurements, timesteps=20, verbose=False):
    measurement_model = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    measurement_noise = 0.1 * np.eye(2)

    settings = Config(detection_probability=0.9,
                      survival_probability=0.9,
                      prune_threshold_poisson=0.01,
                      prune_threshold_global_hypo=-10,
                      prune_threshold_targets=-30,
                      gating_distance=15,
                      birth_gating_distance=10,
                      motion_model='CV',
                      poisson_states_model_name='noob',
                      measurement_model=measurement_model,
                      measurement_noise=measurement_noise,
                      birth_weight=0.9,
                      max_nof_global_hypos=200,
                      min_new_nof_global_hypos=2,
                      max_new_nof_global_hypos=50)

    pmbm = PMBM(settings)

    for timestep in range(min(timesteps, len(measurements))):
        if not verbose: print('k = {}/{} | '.format(timestep + 1, len(measurements)), end='')
        if verbose: print('---\nIteration k = {}'.format(timestep + 1))
        start_total_time = time.time()

        # PREDICTION
        #    print("------------------ PREDICTION: {}------------------".format(timestep))
        start_time = time.time()
        pmbm.predict()
        # pmbm.plot_targets(true_trajectories, measurements, timestep, 'prediction')
        prediction_time = round((time.time() - start_time) / (1e-6), 1)

        # UPDATE
        #    print("------------------ UPDATE: {}------------------".format(timestep))
        start_time = time.time()
        pmbm.update(measurements[timestep])
        update_time = round((time.time() - start_time) / (1e-6), 1)

        start_time = time.time()
        pmbm.target_estimation()
        estimation_time = round((time.time() - start_time) / (1e-6), 1)

        start_time = time.time()
        pmbm.reduction()
        prune_sth_time = round((time.time() - start_time) / (1e-6), 1)

        total_time = round((time.time() - start_total_time), 4)

        if verbose: print(
            '\tTotal time:\t {}\ts \n\t\t \tPrediction:\t {}\t\u03BCs \n\t\t \tUpdate:\t\t {}\t\u03BCs \n\t\t \tEstim.:\t\t {}\t\u03BCs \n\t\t \tPrune:\t\t {}\t\u03BCs \n\t\t \t'.format(
                total_time, prediction_time, update_time, estimation_time, prune_sth_time))
        logger.log_data(pmbm, total_time=total_time, measurements=measurements[timestep],
                        true_states=np.array(true_trajectories)[:, timestep], verbose=verbose)

    return pmbm


