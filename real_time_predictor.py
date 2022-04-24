# libraries imported
import time

import numpy as np
from datetime import datetime
import concurrent.futures
import os


# functions that apply to both simulated and real tremor
import functions.feature_handler as fh
import functions.data_handler as dh
import functions.evaluator as eva
import functions.optimiser as op
import functions.plotter as plt
# buffer class to simulate a real data buffer
from classes.buffer import Buffer

np.set_printoptions(threshold=50)  # shortens long arrays in the console window


def main(FILE_NAME, model_type):
    """ Constants """
    TIME_PERIOD = 1 / 250  # a sample is recorded every 0.004 seconds
    N_SAMPLES = 500  # more samples = more accuracy but slower speed

    # reads data into memory and filters it
    data = dh.read_data(FILE_NAME, 200, 3000)  # real tremor data (t, x, y, z, grip force)
    motion = [data[1], data[2], data[3]]

    with concurrent.futures.ProcessPoolExecutor() as exe:
        """ Buffer filling phase """
        # returns [motion_buffer, label_buffer, reading_times, filtering_time]
        buffer_fill_results = exe.map(
            fill_buffers,
            motion,
            [N_SAMPLES, N_SAMPLES, N_SAMPLES],
            [TIME_PERIOD, TIME_PERIOD, TIME_PERIOD]
        )
        # binds results to variables
        motion_buffer = []
        label_buffer = []
        reading_times = []
        filtering_times = []
        for result in buffer_fill_results:
            motion_buffer.append(result[0])
            label_buffer.append(result[1])
            reading_times.append(result[2])
            filtering_times.append(result[3])

        """ Training and tuning phase """
        # returns [regression, horizon, training_time]
        training_results = exe.map(
            train_model,
            motion_buffer,
            label_buffer,
            [model_type, model_type, model_type],
            [TIME_PERIOD, TIME_PERIOD, TIME_PERIOD]
        )
        # binds results to variables
        regression = []
        horizon = []
        training_times = []
        for result in training_results:
            regression.append(result[0])
            horizon.append(result[1])
            training_times.append(result[2])

        """ Prediction phase """
        # skips all the samples being 'streamed' while the model was trained
        prediction_start = N_SAMPLES + round(np.max(training_times) / TIME_PERIOD)  # index must be an integer
        print("Predictions start at index:", prediction_start)
        BUFFER_LENGTH = 10
        # returns [total_predictions, predicting_times, wait_time]
        prediction_results = exe.map(
            predict_outputs,
            motion,
            regression,
            horizon,
            [prediction_start, prediction_start, prediction_start],
            [BUFFER_LENGTH, BUFFER_LENGTH, BUFFER_LENGTH],
            [TIME_PERIOD, TIME_PERIOD, TIME_PERIOD]
        )
        # binds results to variables
        total_predictions = []
        predicting_times = []
        wait_times = []
        for result in prediction_results:
            total_predictions.append(result[0])
            predicting_times.append(result[1])
            wait_times.append(result[2])

        """ Evaluation phase """
        times = [reading_times, filtering_times, training_times, predicting_times, wait_times]
        start_index = prediction_start + BUFFER_LENGTH
        evaluate_model(times, data, start_index, total_predictions, TIME_PERIOD)


# fills all buffers with data (in the beginning)
def fill_buffers(data, N_SAMPLES, TIME_PERIOD, prediction=False):
    motion_buffer = Buffer([], N_SAMPLES)
    label_buffer = Buffer([], N_SAMPLES)

    print("\nFilling buffer...")
    reading_times = []
    for i in range(N_SAMPLES):
        start_time = datetime.now()

        # buffer is updated
        motion_buffer.add(data[i])

        end_time = datetime.now()
        reading_time = (end_time - start_time).total_seconds()
        # ensures that every iteration 'waits' for the next sample to be streamed
        if reading_time < TIME_PERIOD:
            reading_time = TIME_PERIOD  # should at least 0.004s (sample rate)
        reading_times.append(reading_time)

    # allows skipping of filtering when filling buffer during prediction phase
    if not prediction:
        # generates the labels and normalises the buffers
        start_time = datetime.now()

        # filter is approximately the voluntary motion (label)
        label_buffer.content = motion_buffer.filter(TIME_PERIOD)

        end_time = datetime.now()
        # measures time taken for each iteration
        filtering_time = (end_time - start_time).total_seconds()

        print("\nDone!\n")
        return motion_buffer, label_buffer, reading_times, filtering_time
    else:
        print("\nDone!\n")
        return motion_buffer, reading_times


# trains and tunes a regression model (SVM)
def train_model(motion_buffer, label_buffer, model_type, TIME_PERIOD):
    start_time = datetime.now()

    # calculates the features in a separate function
    [features, horizon] = fh.gen_features(TIME_PERIOD, motion_buffer.normalise(), label_buffer.normalise())

    # SVM with rbf kernel
    print("Tuning...")
    # reformats the features for fitting the model (numpy array)
    features = np.vstack(features).T
    # tunes and trains the regression model
    [regression, hyperparameters] = op.tune_model(features, label_buffer.normalise(), model_type)
    print("Done!")
    print("\nHyperparameters:", hyperparameters)
    print("Horizon value:", horizon)  # prints the optimised values

    end_time = datetime.now()
    # measures time taken for training the model
    training_time = (end_time - start_time).total_seconds()

    return regression, horizon, training_time


# predicts outputs using an already trained regression model (SVM)
def predict_outputs(motion, regression, horizon, prediction_start, buffer_length, TIME_PERIOD):
    total_predictions = []
    predicting_times = []

    # skips data past prediction_start (time spent training the model)
    motion = motion[prediction_start:]

    # fills buffer in prediction mode (no label generation)
    [motion_buffer, reading_times] = fill_buffers(motion, buffer_length, TIME_PERIOD, True)
    label_buffer = Buffer([], buffer_length)

    i = buffer_length  # skips all data already added to the buffer
    index_step = 1  # no skipping in the beginning
    while i < len(motion):
        start_time = datetime.now()

        # loop allows missed data to be saved to buffer
        for j in range(index_step, 0, -1):
            motion_buffer.add(motion[i - j])
        # motion is filtered for denormalisation
        label_buffer.content = motion_buffer.filter(TIME_PERIOD)

        # generates features out of the data in the buffer
        features = fh.gen_features(TIME_PERIOD, motion_buffer.normalise(), horizon=horizon)
        # gets midpoints and spreads to denormalise the predictions
        [midpoint, sigma] = label_buffer.get_data_attributes()

        # reformats the features for fitting the model (numpy array)
        features = np.vstack(features).T
        # predicts the voluntary motion and denormalises it to the correct scale
        prediction = fh.denormalise(regression.predict(features), midpoint, sigma)
        # selects and saves only the new predictions to an external array for evaluation
        new_predictions = prediction[len(prediction) - index_step:len(prediction)]
        for value in new_predictions:
            total_predictions.append(value)

        end_time = datetime.now()
        predict_time = (end_time - start_time).total_seconds()
        # limits prediction time to the sample time period (can't be faster than input 'stream')
        if predict_time < TIME_PERIOD:
            predict_time = TIME_PERIOD
        # measures time taken for predicting
        predicting_times.append(predict_time)

        # ensures the last sample is not missed
        if (i + index_step) > len(motion) and i != (len(motion) - 1):
            index_step = len(motion) - i - 1
        else:
            # skips all the samples being 'streamed' while the program performed predictions
            index_step = round(predict_time / TIME_PERIOD)  # must be an integer
        i += index_step

    return total_predictions, predicting_times, sum(reading_times)


# evaluates model: Prints performance + accuracy and plots graphs
def evaluate_model(times, data, start_index, total_predictions, TIME_PERIOD):
    print("\nResults\n==============================================")  # separates results from other messages
    reading_times = times[0]
    filtering_time = times[1]
    training_time = times[2]
    predicting_times = times[3]
    wait_time = times[4]

    # prints time based performance results
    total_reading_time = [sum(reading_times[0]), sum(reading_times[1]), sum(reading_times[2])]
    total_filtering_time = filtering_time
    avg_predicting_times = [np.mean(predicting_times[0]), np.mean(predicting_times[1]), np.mean(predicting_times[2])]
    max_predicting_times = [np.max(predicting_times[0]), np.max(predicting_times[1]), np.max(predicting_times[2])]
    min_predicting_times = [np.min(predicting_times[0]), np.min(predicting_times[1]), np.min(predicting_times[2])]
    avg_index_skipped = np.round(np.divide(avg_predicting_times, TIME_PERIOD))
    max_index_skipped = np.round(np.divide(max_predicting_times, TIME_PERIOD))
    total_prediction_time = [sum(predicting_times[0]), sum(predicting_times[1]), sum(predicting_times[2])]
    print(
        "\nTotal time filling buffer:", np.max(total_reading_time),
        "\nTotal time filtering data:", np.max(total_filtering_time),
        "\nTotal time taken during training/tuning:", np.max(training_time),
        "\nAverage time taken to predict voluntary motion:", np.max(avg_predicting_times),
        "\nAverage samples per prediction loop [X, Y, Z]:", avg_index_skipped,
        "\nMaximum time taken for a prediction [X, Y, Z]:", max_predicting_times,
        "\nMinimum time taken for a prediction [X, Y, Z]:", min_predicting_times,
        "\nMaximum samples per prediction loop [X, Y, Z]:", max_index_skipped,
        "\nTotal prediction time:",
        np.max(total_prediction_time), "+",
        np.max(wait_time), "=",
        np.max(np.add(total_prediction_time, wait_time))
    )

    # truncates the data to the same length as the predictions
    motion = [data[1][start_index:], data[2][start_index:], data[3][start_index:]]
    filtered_motion = []
    overall_accuracy = []
    # calculates the labels and accuracy of the truncated data
    for i in range(len(motion)):
        filtered_motion.append(dh.filter_data(motion[i], TIME_PERIOD))
        overall_accuracy.append(eva.calc_accuracy(filtered_motion[i], total_predictions[i]))
    # prints the accuracies of the overall voluntary motion (after completion)
    print(
        "\nOverall accuracy",
        "\nX [R2, NRMSE]: [" + str(overall_accuracy[0][0]) + "%" + ", " + str(overall_accuracy[0][1]) + "]",
        "\nY [R2, NRMSE]: [" + str(overall_accuracy[1][0]) + "%" + ", " + str(overall_accuracy[1][1]) + "]",
        "\nZ [R2, NRMSE]: [" + str(overall_accuracy[2][0]) + "%" + ", " + str(overall_accuracy[2][1]) + "]"
    )

    # gets the tremor component by subtracting from the voluntary motion
    actual_tremor = []
    predicted_tremor = []
    overall_tremor_accuracy = []
    for i in range(len(motion)):
        actual_tremor.append(np.subtract(motion[i], filtered_motion[i]))
        predicted_tremor.append(np.subtract(motion[i], total_predictions[i]))
        overall_tremor_accuracy.append(eva.calc_tremor_accuracy(motion[i], total_predictions[i], filtered_motion[i]))
    tremor_error = np.subtract(actual_tremor, predicted_tremor)
    # prints the accuracies of the overall tremor component (after completion)
    print(
        "\nOverall tremor accuracy",
        "\nX [R2, NRMSE]: ["
        + str(overall_tremor_accuracy[0][0]) + "%" + ", " + str(overall_tremor_accuracy[0][1]) +
        "]",
        "\nY [R2, NRMSE]: ["
        + str(overall_tremor_accuracy[1][0]) + "%" + ", " + str(overall_tremor_accuracy[1][1]) +
        "]",
        "\nZ [R2, NRMSE]: ["
        + str(overall_tremor_accuracy[2][0]) + "%" + ", " + str(overall_tremor_accuracy[2][1]) +
        "]"
    )

    # puts regression model data in a list
    model_data = [
        [motion[0], filtered_motion[0], total_predictions[0], "X motion (mm)"],
        [motion[1], filtered_motion[1], total_predictions[1], "Y motion (mm)"],
        [motion[2], filtered_motion[2], total_predictions[2], "Z motion (mm)"]
    ]
    model_axes_labels = ["Original signal", "Filtered output", "Predicted output"]
    # puts the tremor component data in a list
    tremor_data = [
        [actual_tremor[0], predicted_tremor[0], tremor_error[0], "X motion (mm)"],
        [actual_tremor[1], predicted_tremor[1], tremor_error[1], "Y motion (mm)"],
        [actual_tremor[2], predicted_tremor[2], tremor_error[2], "Z motion (mm)"]
    ]
    tremor_axes_labels = ["Actual tremor", "Predicted tremor", "Tremor error"]

    t = np.array(data[0], dtype='f') * TIME_PERIOD  # samples are measured at a rate of 250Hz
    plt.plot_model(t[start_index:], model_data, model_axes_labels)  # plots SVR model
    plt.plot_model(t[start_index:], tremor_data, tremor_axes_labels)  # plots the tremor components


if __name__ == '__main__':
    # model = "SVM"
    model = "Random Forest"

    # finds the directory
    folder_name = "/Surgeon Tracing/"
    directory_name = "C:/Users/Abdul/OneDrive - Newcastle University/Stage 3/Obsidian Vault/EEE3095-7 Individual Project and Dissertation/Tremor ML/data/" + folder_name[1:]
    directory = os.fsencode(directory_name)

    # allows a specific file to be selected instead of an entire directory
    override_file = "/real_tremor_data.csv"
    if len(override_file) > 0:
        main("./data" + override_file, model)
    else:
        # puts all txt files' names in a list
        file_names = []
        for file in os.listdir(directory):
            file_names.append(os.fsdecode(file))
        # runs predictor algorithm for each dataset
        for file_name in file_names:
            main("./data" + folder_name + file_name, model)
