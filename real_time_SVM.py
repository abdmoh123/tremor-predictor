# libraries imported
import time
import numpy as np
from scipy import signal
from datetime import datetime


# functions that apply to both simulated and real tremor
import functions.feature_handler as fh
import functions.data_handler as dh
import functions.evaluator as eva
import functions.optimiser as op
import functions.plotter as plt

np.set_printoptions(threshold=50)  # shortens long arrays in the console window


def main():
    file_name = "./data/real_tremor_data.csv"
    time_period = 1 / 250  # a sample is recorded every 0.004 seconds

    # reads data into memory and filters it
    data = dh.read_data(file_name, 200, 300)  # real tremor data (t, x, y, z, grip force)
    t = np.array(data[0], dtype='f') * time_period  # samples are measured at a rate of 250Hz

    # a buffer holding a specified number of motion (+ filtered motion) samples
    no_samples = 50  # more samples = more accuracy but slower speed
    motion_buffer = [[], [], []]  # 2D list holding each axis

    time_delay = []
    accuracies = [[], [], []]
    tremor_accuracies = [[], [], []]
    for i in range(len(data[0])):
        start_time = datetime.now()

        print("\nProgress:\n", i+1, "/", len(data[0]), "\n")  # prints progress (for testing purposes)

        current_motion = [data[1][i], data[2][i], data[3][i]]
        # buffers are updated with new values
        motion_buffer = add_to_buffer(current_motion, motion_buffer, no_samples)

        if len(motion_buffer[0]) == no_samples:
            label_buffer = []
            normalised_label_buffer = []
            current_label = []
            current_normal_label = []

            normalised_motion_buffer = []
            motion_means = []
            motion_sigmas = []

            # generates the labels and normalises the buffers
            for j in range(len(motion_buffer)):
                label_buffer.append(filter_data(motion_buffer[j], time_period))

                [n_motion, mean, sigma] = fh.normalise(motion_buffer[j], True)
                normalised_motion_buffer.append(n_motion)
                normalised_label_buffer.append(fh.normalise(label_buffer[j]))
                # means and standard deviations are saved for denormalisation later
                motion_means.append(mean)
                motion_sigmas.append(sigma)

                current_label.append(label_buffer[j][len(label_buffer)])
                current_normal_label.append(normalised_label_buffer[j][len(normalised_label_buffer)])

            # calculates the features in a separate function
            [features, horizon] = fh.gen_features(t, motion_buffer, label_buffer)

            # SVM with rbf kernel (x axis)
            regression = []
            hyperparameters = []
            print("Tuning...")
            for j in range(len(features)):
                # reformats the features for fitting the model (numpy array)
                axis_features = np.vstack(features[j]).T
                # tunes and trains the regression model
                regression.append(op.tune_model(axis_features, label_buffer[j]))
                hyperparameters.append(regression[j].best_params_)
            print("Done!")
            print("\nHyperparameters [{x}, {y}, {z}]:", hyperparameters)
            print("Horizon values [x, y, z]:", horizon)  # prints the optimised values

            # predicts intended motion using the original data as an input (scaled to intended motion)
            prediction = []
            for j in range(len(features)):
                # reformats the features for fitting the model (numpy array)
                axis_features = np.vstack(features[j]).T
                prediction.append(
                    fh.denormalise(regression[j].predict(axis_features), motion_means[j], motion_sigmas[j])
                )

            # calculates and prints the R2 score and normalised RMSE of the model (including tremor component)
            for j in range(len(label_buffer)):
                accuracies[j].append(eva.calc_accuracy(label_buffer[j], prediction[j]))
                tremor_accuracies[j].append(eva.calc_tremor_accuracy(motion_buffer[j], prediction[j], label_buffer[j]))
        end_time = datetime.now()
        # measures time taken for each iteration
        iteration_time = (end_time - start_time).total_seconds()
        time_delay.append(iteration_time)
        # ensures that every iteration 'waits' for the next sample to be streamed
        if iteration_time < time_period:
            time.sleep(time_period - iteration_time)

    print("\nResults\n==============================================")  # separates results from other messages

    r2_scores = []
    nrmse = []
    # fills and formats lists with data
    for i in range(len(accuracies)):
        accuracies[i] = np.array(accuracies[i]).T.tolist()
        tremor_accuracies[i] = np.array(tremor_accuracies[i]).T.tolist()
        r2_scores.append([accuracies[i][0], tremor_accuracies[i][0]])
        nrmse.append([accuracies[i][1], tremor_accuracies[i][1]])
    # prints accuracy scores
    print("\nR2 scores [[motion], [tremor]]\nX:", r2_scores[0], "\nY:", r2_scores[1], "\nZ:", r2_scores[2])
    print("\nNormalised RMS error [[motion], [tremor]]\nX:", nrmse[0], "\nY:", nrmse[1], "\nZ:", nrmse[2])
    # prints performance data
    print("\nTime taken for each iteration:\n", time_delay)
    # prints averages of results
    print(
        "\nAverage R2 scores",
        "\nX [motion, tremor]: [", np.mean(r2_scores[0][0]), "%\t", np.mean(r2_scores[0][1]), "% ]",
        "\nY [motion, tremor]: [", np.mean(r2_scores[1][0]), "%\t", np.mean(r2_scores[1][1]), "% ]",
        "\nZ [motion, tremor]: [", np.mean(r2_scores[2][0]), "%\t", np.mean(r2_scores[2][1]), "% ]"
    )
    print(
        "\nAverage normalised RMS errors",
        "\nX [motion, tremor]: [", np.mean(nrmse[0][0]), "\t", np.mean(nrmse[0][1]), "]",
        "\nY [motion, tremor]: [", np.mean(nrmse[1][0]), "\t", np.mean(nrmse[1][1]), "]",
        "\nZ [motion, tremor]: [", np.mean(nrmse[2][0]), "\t", np.mean(nrmse[2][1]), "]"
    )
    # filling the buffer takes approximately no time
    print("\nAverage time taken after filling buffer:", np.mean(time_delay[(no_samples - 1):]), "s")

    accuracies_labels = [
        ["Motion (x)", "Tremor component (x)"],
        ["Motion (y)", "Tremor component (y)"],
        ["Motion (z)", "Tremor component (z)"]
    ]
    plt.plot_accuracies(
        data[0][(no_samples - 1):],
        r2_scores,
        accuracies_labels,
        "Iteration",
        "R2 accuracy (%)"
    )
    plt.plot_accuracies(
        data[0][(no_samples - 1):],
        nrmse,
        accuracies_labels,
        "Iteration",
        "Normalised error"
    )
    # filling buffer takes approximately no time so it was removed to scale important data better
    plt.plot_data(
        data[0][(no_samples - 1):],
        [[time_delay[(no_samples - 1):], "Delay per iteration"]],
        "Iteration",
        "Time taken (s)"
    )


# filters the input data to estimate the intended movement
def filter_data(data, time_period):
    nyquist = 1 / (2 * time_period)
    cut_off = 5 / nyquist

    # zero phase filter is used to generate the labels (slow but very accurate)
    [b, a] = signal.butter(2, cut_off, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    return np.ndarray.tolist(filtered_data)  # converts np array to list


# simulates a buffer with motion data being streamed in
def add_to_buffer(motion, buffer, max_length):
    # appends new data to the end of the list
    for i in range(len(motion)):
        buffer[i].append(motion[i])  # [[X motion...], [Y motion...], [Z motion...]]
    # prevents the buffer from exceeding the maximum length
    if len(buffer[0]) > max_length:
        buffer = [
            buffer[0][len(buffer[0]) - max_length:],
            buffer[1][len(buffer[1]) - max_length:],
            buffer[2][len(buffer[2]) - max_length:]
        ]
    return buffer


if __name__ == '__main__':
    main()
