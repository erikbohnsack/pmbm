from deap import base, creator, tools
from deap import algorithms
import numpy as np
from operator import attrgetter
from pmbm.pmbm import PMBM
from pmbm.config import Config
from utils.mot_metrics import MotCalculator
import motmetrics as mm
from utils.eval_metrics import GOSPA
from data_utils import kitti_stuff
import platform
import random
import time


class Params:
    def __init__(self):
        self.measurement_var_xy = (0.01, 5)
        self.measurement_var_psi = (1, 5)

        self.poisson_vx = (1, 10)
        self.poisson_vy = (1, 10)
        self.poisson_v = (1, 10)
        self.poisson_d = (1, 15)

        self.sigma_v = (1, 15)
        self.sigma_d = (1, 10)
        self.sigma_phi = (1, 10)


def ga():
    ngen = 50
    n_ind = 20
    print("Running Genetic Algorithm. # Gen: {}, # Individuals: {}".format(ngen, n_ind))
    toolbox = base.Toolbox()
    # CONSTRAINT_PENALTY = 10000
    p = Params()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.decorate("mate", checkBounds(p))
    toolbox.decorate("mutate", checkBounds(p))

    toolbox.register("evaluate", evalFct)
    # toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, CONSTRAINT_PENALTY))

    toolbox.register("attr_meas_var_xy", random.uniform, p.measurement_var_xy[0], p.measurement_var_xy[1])
    toolbox.register("attr_meas_var_psi", random.uniform, p.measurement_var_psi[0], p.measurement_var_psi[1])
    toolbox.register("attr_poisson_vx", random.uniform, p.poisson_vx[0], p.poisson_vx[1])
    toolbox.register("attr_poisson_vy", random.uniform, p.poisson_vy[0], p.poisson_vy[1])
    toolbox.register("attr_poisson_v", random.uniform, p.poisson_v[0],
                     p.poisson_v[1])
    toolbox.register("attr_poisson_d", random.uniform, p.poisson_d[0],
                     p.poisson_d[1])
    toolbox.register("attr_sigma_v", random.uniform, p.sigma_v[0],
                     p.sigma_v[1])
    toolbox.register("attr_sigma_d", random.uniform, p.sigma_d[0], p.sigma_d[1])
    toolbox.register("attr_sigma_phi", random.uniform, p.sigma_phi[0], p.sigma_phi[1])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_meas_var_xy, toolbox.attr_meas_var_psi,
                      toolbox.attr_poisson_vx, toolbox.attr_poisson_vy,
                      toolbox.attr_poisson_v, toolbox.attr_poisson_d,
                      toolbox.attr_sigma_v, toolbox.attr_sigma_d,
                      toolbox.attr_sigma_phi), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=n_ind)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen)
    best = max(pop, key=attrgetter("fitness"))
    print(best, best.fitness)


def evalFct(individual):
    """ Evaluation function. Should return ."""
    try:
        result = get_score_4_ind(individual)
    except:
        return 0.,
    return result,


def feasible(individual):
    """Feasibility function for the individual. Returns True if feasible False
    otherwise."""
    boolie = [False] * len(individual)
    p = Params()
    i = 0

    for key, value in p.__dict__.items():
        if value[0] <= individual[i] <= value[1]:
            boolie[i] = True
        i += 1

    if boolie:
        return True
    return False


def checkBounds(params):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                i = 0
                for key, value in params.__dict__.items():
                    if child[i] > value[1]:
                        child[i] = value[1]
                    elif child[i] < value[0]:
                        child[i] = value[0]
                    if i >= 13 and isinstance(child[i], float):
                        child[i] = int(round(child[i]))
                    i += 1
                assert i == len(child)
            return offspring

        return wrapper

    return decorator


def get_score_4_ind(individual):
    tic = time.time()
    if platform.system() == 'Darwin':
        root = '/Users/erikbohnsack/data/'
    else:
        root = '/home/mlt/data'
    kitti = kitti_stuff.Kitti(ROOT=root, split='training')

    config = ind_2_config(individual)

    state_dims = 2
    tot_score = 0
    # sequence_ids = random.sample(range(0, 20), 2)
    sequence_ids = [3, 5, 7]
    max_frames = 100
    for seq_id in sequence_ids:
        pmbm = PMBM(config)
        acc = MotCalculator(seq_id, path_to_data=root)
        kitti.lbls = kitti.load_labels(seq_id)
        imud = kitti.load_imu(seq_id)
        gospa_score = 0
        n_unique_id = 0
        for frame_idx in range(min(len(kitti.lbls), max_frames)):
            max_track_id = max([lbl.track_id for lbl in kitti.lbls[frame_idx]])
            if max_track_id > n_unique_id:
                n_unique_id = max_track_id
            measurements, classes = kitti.get_measurements(frame_idx, measurement_dims=3, p_missed=0.08, p_clutter=0.02, p_mutate=0.7)
            pmbm.run(measurements, classes, imud[frame_idx], frame_idx, verbose=False)

            ground_truths = np.array([], dtype=np.float).reshape(0, 2)
            for l in kitti.lbls[frame_idx]:
                if l.type[0] == 'DontCare':
                    continue

                x_pos = l.location[0]
                z_pos = l.location[2]
                ground_truths = np.vstack((ground_truths, np.array([[x_pos, z_pos]])))

            estimated_states = np.array([], dtype=np.float).reshape(0, 2)
            for et in pmbm.estimated_targets:
                _state = et['single_target'].state
                estimated_states = np.vstack((estimated_states, _state[0:state_dims].reshape(1, state_dims)))
            acc.calculate(pmbm.estimated_targets, frame_idx)
            gospa_score += GOSPA(ground_truths, estimated_states, p=1, c=100, alpha=2., state_dim=state_dims)
        mh = mm.metrics.create()
        gospa_score /= frame_idx
        summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'mostly_tracked', 'mostly_lost',
                                           'num_false_positives', 'num_switches', 'num_fragmentations'], name=str(seq_id))
        summary.at[str(seq_id), 'mostly_tracked'] /= n_unique_id
        summary.at[str(seq_id), 'mostly_lost'] /= n_unique_id
        summary.at[str(seq_id), 'num_false_positives'] /= n_unique_id
        summary.at[str(seq_id), 'num_switches'] /= n_unique_id
        summary.at[str(seq_id), 'num_fragmentations'] /= n_unique_id
        mota = summary['mota'].values
        motp = summary['motp'].values
        num_switches = summary['num_switches'].values
        num_false_positives = summary['num_false_positives'].values
        num_fragmentations = summary['num_fragmentations'].values

        if num_false_positives == 0:
            num_false_positives = 1
        if num_switches == 0:
            num_switches = 1
        if num_fragmentations == 0:
            num_fragmentations = 1
        tot_score += 1 / gospa_score + mota + 1 - motp + 0.5 / num_switches + 0.5 / num_false_positives + 0.5 / num_fragmentations
    toc = time.time() - tic
    print("Individual evaluated, score: {}. Time taken: {}\nIndivudial:{}".format(tot_score, toc, individual))
    return tot_score


def ind_2_config(individual):
    config = Config(config_name='Mixed-GA',
                    motion_model='Mixed',
                    poisson_states_model_name='uniform-mixed',
                    filter_class='Mixed',
                    measurement_var_xy=individual[0],
                    measurement_var_psi=individual[1],
                    poisson_vx=individual[2],
                    poisson_vy=individual[3],
                    poisson_v=individual[4],
                    poisson_d=individual[5],
                    sigma_phi=individual[6],
                    sigma_v=individual[7],
                    sigma_d=individual[8])
    return config
