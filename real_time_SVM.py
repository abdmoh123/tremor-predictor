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
    data = dh.read_data(file_name, 200, 4000)  # real tremor data (t, x, y, z, grip force)
    t = np.array(data[0], dtype='f') * time_period  # samples are measured at a rate of 250Hz

    # a buffer holding a specified number of motion (+ filtered motion) samples
    no_samples = 750  # more samples = more accuracy but slower speed
    motion_buffer = [[], [], []]  # 2D list holding each axis
    normalised_motion_buffer = []
    label_buffer = []
    normalised_label_buffer = []

    accuracies = [[], [], []]
    tremor_accuracies = [[], [], []]
    total_predictions = [[], [], []]

    reading_times = []
    filtering_times = []
    training_times = []
    predicting_times = []

    for i in range(no_samples):
        start_time = datetime.now()

        print("\nProgress:\n", i + 1, "/", len(data[0]), "\n")  # prints progress (for testing purposes)

        current_motion = [data[1][i], data[2][i], data[3][i]]
        # buffer is updated
        motion_buffer = add_to_buffer(current_motion, motion_buffer, no_samples)

        end_time = datetime.now()
        # ensures that every iteration 'waits' for the next sample to be streamed
        if reading_time < time_period:
            reading_times = time_period  # should at least 0.004s (sample rate)
        else:
            reading_time = (end_time - start_time).total_seconds()  # measures time taken for each iteration
        reading_times.append(reading_time)

    # generates the labels and normalises the buffers
    for i in range(len(motion_buffer)):
        start_time = datetime.now()

        # filter is approximately the voluntary motion (label)
        label_buffer.append(filter_data(motion_buffer[i], time_period))

        # buffers are normalised for use in training
        normalised_motion_buffer.append(fh.normalise(motion_buffer[i]))
        normalised_label_buffer.append(fh.normalise(label_buffer[i]))

        end_time = datetime.now()
        # measures time taken for each iteration
        filtering_times.append((end_time - start_time).total_seconds())

    start_time = datetime.now()

    # calculates the features in a separate function
    [features, horizon] = fh.gen_features(t, normalised_motion_buffer, normalised_label_buffer)

    # SVM with rbf kernel (x axis)
    regression = []
    hyperparameters = []
    print("Tuning...")
    for j in range(len(features)):
        # reformats the features for fitting the model (numpy array)
        axis_features = np.vstack(features[j]).T
        # tunes and trains the regression model
        regression.append(op.tune_model(axis_features, normalised_label_buffer[j]))
        hyperparameters.append(regression[j].best_params_)
    print("Done!")
    print("\nHyperparameters [{x}, {y}, {z}]:", hyperparameters)
    print("Horizon values [x, y, z]:", horizon)  # prints the optimised values

    end_time = datetime.now()
    # measures time taken for training the model
    training_times.append((end_time - start_time).total_seconds())

    for i in range(no_samples, len(data[0])):
        start_time = datetime.now()

        current_motion = [data[1][i], data[2][i], data[3][i]]
        motion_buffer = add_to_buffer(current_motion, motion_buffer, no_samples)

        # empties the means and standard deviations before normalising the new data
        motion_means = []
        motion_sigmas = []
        normalised_motion_buffer = []
        # buffer is normalised for predicting labels accurately
        for j in range(len(motion_buffer)):
            [n_motion, mean, sigma] = fh.normalise(motion_buffer[j], True)
            normalised_motion_buffer.append(n_motion)
            # means and standard deviations are saved for denormalisation later
            motion_means.append(mean)
            motion_sigmas.append(sigma)

        # generates features out of the data in the buffer
        features = fh.gen_features(t, normalised_motion_buffer, horizon=horizon)
        # predicts intended motion using the original data as an input (scaled to intended motion)
        prediction = []
        for j in range(len(features)):
            # reformats the features for fitting the model (numpy array)
            axis_features = np.vstack(features[j]).T
            # predicts the voluntary motion and denormalises it to the correct scale
            prediction.append(fh.denormalise(regression[j].predict(axis_features), motion_means[j], motion_sigmas[j]))
            total_predictions[j].append(prediction[j][len(prediction) - 1])  # saves latest prediction for evaluation

        print("\nProgress:\n", i + 1, "/", len(data[0]), "\n")  # prints progress (for testing purposes)

        # calculates and prints the R2 score and normalised RMSE of the model (including tremor component)
        for j in range(len(label_buffer)):
            accuracies[j].append(eva.calc_accuracy(label_buffer[j], prediction[j]))
            tremor_accuracies[j].append(eva.calc_tremor_accuracy(motion_buffer[j], prediction[j], label_buffer[j]))

        end_time = datetime.now()
        # measures time taken for predicting
        predicting_times.append((end_time - start_time).total_seconds())

    print("\nResults\n==============================================")  # separates results from other messages

    r2_scores = []
    nrmse = []
    # fills and formats lists with data
    for i in range(len(accuracies)):
        accuracies[i] = np.array(accuracies[i]).T.tolist()
        tremor_accuracies[i] = np.array(tremor_accuracies[i]).T.tolist()
        r2_scores.append([accuracies[i][0], tremor_accuracies[i][0]])
        nrmse.append([accuracies[i][1], tremor_accuracies[i][1]])
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

    total_reading_time = sum(reading_times)
    total_filtering_time = sum(filtering_times)
    total_training_time = sum(training_times)
    avg_prediction_time = np.mean(predicting_times)
    print(
        "\nTotal time filling buffer:", total_reading_time,
        "\nTotal time filtering data:", total_filtering_time,
        "\nTotal time taken during training/tuning:", total_training_time,
        "\nAverage time taken to predict voluntary motion:", avg_prediction_time
    )

    motion = [data[1][no_samples:], data[2][no_samples:], data[3][no_samples:]]
    filtered_motion = []
    for axis in motion:
        filtered_motion.append(filter_data(axis, time_period))
    # puts regression model data in a list
    model_data = [
        [motion[0], filtered_motion[0], total_predictions[0], "X motion (mm)"],
        [motion[1], filtered_motion[1], total_predictions[1], "Y motion (mm)"],
        [motion[2], filtered_motion[2], total_predictions[2], "Z motion (mm)"]
    ]
    model_axes_labels = ["Original signal", "Filtered output", "Predicted output"]

    # gets the tremor component by subtracting from the voluntary motion
    actual_tremor = []
    predicted_tremor = []
    tremor_accuracy = []
    for i in range(len(motion)):
        actual_tremor.append(np.subtract(motion[i], filtered_motion[i]))
        predicted_tremor.append(np.subtract(motion[i], total_predictions[i]))
        # calculates the normalised RMSE of the tremor component
        tremor_accuracy.append(eva.calc_accuracy(actual_tremor[i], predicted_tremor[i]))
    tremor_error = np.subtract(actual_tremor, predicted_tremor)
    # puts the tremor component data in a list
    tremor_data = [
        [actual_tremor[0], predicted_tremor[0], tremor_error[0], "X motion (mm)"],
        [actual_tremor[1], predicted_tremor[1], tremor_error[1], "Y motion (mm)"],
        [actual_tremor[2], predicted_tremor[2], tremor_error[2], "Z motion (mm)"]
    ]
    tremor_axes_labels = ["Actual tremor", "Predicted tremor", "Tremor error"]

    plt.plot_model(t[no_samples:], model_data, model_axes_labels)  # plots SVR model
    plt.plot_model(t[no_samples:], tremor_data, tremor_axes_labels)  # plots the tremor components

    accuracies_labels = [
        ["Motion (x)", "Tremor component (x)"],
        ["Motion (y)", "Tremor component (y)"],
        ["Motion (z)", "Tremor component (z)"]
    ]
    plt.plot_accuracies(
        data[0][no_samples:],
        r2_scores,
        accuracies_labels,
        "Iteration",
        "R2 accuracy (%)"
    )
    plt.plot_accuracies(
        data[0][no_samples:],
        nrmse,
        accuracies_labels,
        "Iteration",
        "Normalised error"
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
