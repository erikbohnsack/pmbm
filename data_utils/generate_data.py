import matplotlib.pyplot as plt
import numpy as np
from utils import motion_models
import random


class GenerateData:
    def __init__(self):
        pass

    def generate_2D_data(initial_states=[np.asarray([[-45], [45], [10], [-5]]), np.asarray([[-40], [-30], [5], [15]])],
                         initial_variance=np.eye(4),
                         motion_noise=0.1,
                         mutation_probability=0.4,
                         measurement_noise=0.4,
                         missed_detection_probability=0.05,
                         clutter_probability=0.1,
                         dT=1,
                         time_steps=20):
        ''' Generating two trajectories in a 2D grid
        Input: -

        Output: true_trajectory[ trajectory, time_step , [x,y,dx,dy] ]
                   measurements[ time_step, measurements, [x,y] ]
        '''

        targets = initial_states
        cv_model = motion_models.ConstantVelocityModel(motion_noise)

        true_trajectories = []
        for i in range(len(targets)):
            true_trajectory = []
            next_state = targets[i]
            true_trajectory.append(next_state)
            for t in range(time_steps):
                next_state, _ = cv_model(next_state, initial_variance, dT=dT)

                # Mutation
                if mutation_probability < random.random():
                    next_state[0] = next_state[0] + random.uniform(-0.05, 0.05) * next_state[0]
                if mutation_probability < random.random():
                    next_state[1] = next_state[1] + random.uniform(-0.05, 0.05) * next_state[1]
                if mutation_probability < random.random():
                    next_state[2] = next_state[2] + random.uniform(-0.05, 0.05) * next_state[2]
                if mutation_probability < random.random():
                    next_state[3] = next_state[3] + random.uniform(-0.05, 0.05) * next_state[3]

                true_trajectory.append(next_state)
            true_trajectories.append(true_trajectory)

        measurements = []
        for t in range(time_steps):
            current_measurements = []
            for i in range(len(targets)):
                x_pos = true_trajectories[i][t][0]
                y_pos = true_trajectories[i][t][1]

                x_pos = (1 - random.uniform(-measurement_noise, measurement_noise)) * x_pos
                y_pos = (1 - random.uniform(-measurement_noise, measurement_noise)) * y_pos

                if missed_detection_probability > random.random():
                    meas = np.array([], dtype=np.float).reshape(0, 2)
                else:
                    meas = np.array((x_pos, y_pos))
                current_measurements.append(meas)

            if clutter_probability > random.random():
                x_pos = (1 - random.uniform(-2 * measurement_noise, 2 * measurement_noise)) * x_pos
                y_pos = (1 - random.uniform(-2 * measurement_noise, 2 * measurement_noise)) * y_pos
                meas = np.array((x_pos, y_pos))
                current_measurements.append(meas)

            measurements.append(current_measurements)

        return true_trajectories, measurements




    def plot_generated_data(true_trajectories, measurements):
        plt.figure()
        for tt in true_trajectories:
            _x = [x[0] for x in tt]
            _y = [x[1] for x in tt]
            plt.subplot(1, 1, 1)
            plt.plot(_x, _y, 'o-')
            plt.title('A tale of 2 targets')
            plt.ylabel('y')
            plt.xlabel('x')
            plt.grid()

        for t, current_measurements in enumerate(measurements):
            for meas in current_measurements:
                try:
                    _x = meas[0]
                    _y = meas[1]
                    plt.subplot(1, 1, 1)
                    plt.plot(_x, _y, 'ks')
                except:
                    continue

        plt.show()



