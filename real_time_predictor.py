# libraries imported
import math
import numpy as np
from scipy import interpolate
from datetime import datetime
import concurrent.futures


# functions that apply to both simulated and real tremor
import functions.feature_handler as fh
import functions.data_handler as dh
import functions.evaluator as eva
import functions.optimiser as op
import functions.plotter as pltr
# buffer class to simulate a real data buffer
from classes.buffer import Buffer

np.set_printoptions(threshold=50)  # shortens long arrays in the console window


def start_predictor(FILE_NAME, MODEL_TYPE):
    """ Constants """
    TIME_PERIOD = 1 / 250  # a sample is recorded every 0.004 seconds
    N_SAMPLES = 500  # more samples = more accuracy but slower speed (training buffer length)
    BUFFER_LENGTH = 10  # prediction buffer length

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

        """ Tuning phase """
        # returns [regression, horizon, training_time]
        training_results = exe.map(
            train_model,
            motion_buffer,
            label_buffer,
            [MODEL_TYPE, MODEL_TYPE, MODEL_TYPE]
        )
        # binds results to variables
        regression = []
        hyperparameters = []
        horizon = []
        training_times = []
        for result in training_results:
            regression.append(result[0])
            hyperparameters.append(result[1])
            horizon.append(result[2])
            training_times.append(result[3])
        print("\nHyperparameters:", hyperparameters)  # prints hyperparameters
        print("Horizon values:", horizon)  # prints the optimised values

        """ Prediction phase """
        # skips all the samples being 'streamed' while the model was trained
        prediction_start = N_SAMPLES + round(np.max(training_times) / TIME_PERIOD)  # index must be an integer
        print("Predictions start at index:", prediction_start)
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
        return evaluate_model(times, data, hyperparameters, start_index, total_predictions, TIME_PERIOD)


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

        print("Done!")
        return motion_buffer, label_buffer, reading_times, filtering_time
    else:
        print("Done!")
        return motion_buffer, reading_times


# trains and tunes a regression model (SVM)
def train_model(motion_buffer, label_buffer, model_type):
    start_time = datetime.now()

    # calculates the features in a separate function
    [features, horizon] = fh.gen_features(motion_buffer.normalise(), label_buffer.normalise())

    # SVM with rbf kernel
    print("\nTuning...")
    # reformats the features for fitting the model (numpy array)
    features = np.vstack(features).T
    # tunes and trains the regression model
    [regression, hyperparameters] = op.tune_model(features, label_buffer.normalise(), model_type)
    print("Done!")

    end_time = datetime.now()
    # measures time taken for training the model
    training_time = (end_time - start_time).total_seconds()

    return regression, hyperparameters, horizon, training_time


# predicts outputs using an already trained regression model (SVM)
def predict_outputs(motion, regression, horizon, prediction_start, buffer_length, TIME_PERIOD):
    total_predictions = []
    predicting_times = []

    # skips data past prediction_start (time spent training the model)
    motion = motion[prediction_start:]

    # fills buffer in prediction mode (no label generation)
    [motion_buffer, reading_times] = fill_buffers(motion, buffer_length, TIME_PERIOD, True)

    # buffer for a linear butterworth (not zero-phase) IIR filter is prepared
    filter_buffer = Buffer(motion_buffer.content, 3000)
    # delay of IIR filter is calculated and printed
    [freq, samples] = filter_buffer.get_filter_delay(TIME_PERIOD)
    filter_delay = samples[len(samples) - 1] / freq[len(freq) - 1]
    print("Phase delay:", str(filter_delay) + "s", round(samples[len(samples) - 1]), "samples behind")

    print("\nPredicting...")

    i = buffer_length  # skips all data already added to the buffer
    index_step = 0  # no skipping in the beginning
    while i < len(motion):
        start_time = datetime.now()

        # loop allows missed data to be saved to buffer
        for j in range(index_step, 0, -1):
            # +1 ensures that the current motion is added
            motion_buffer.add(motion[i - j + 1])
            filter_buffer.add(motion[i - j + 1])

        # generates features out of the data in the buffer
        features = np.vstack(fh.gen_features(fh.shift(motion_buffer.normalise()), horizon=horizon)).T
        [midpoint, sigma] = fh.get_norm_attributes(
            filter_buffer.filter(TIME_PERIOD, False)[len(filter_buffer.content) - buffer_length:]
        )
        # predicts the voluntary motion and denormalises it to the correct scale
        prediction = fh.denormalise(regression.predict(features), midpoint, sigma)

        # selects and saves only the new predictions to an external array for evaluation
        if len(prediction) > index_step:
            new_predictions = prediction[len(prediction) - index_step:]
        else:
            new_predictions = prediction
        for value in new_predictions:
            total_predictions.append(value)

        end_time = datetime.now()
        # measures time taken for predicting
        predict_time = (end_time - start_time).total_seconds()
        predicting_times.append(predict_time)

        # skips all the samples being 'streamed' while the program performed predictions
        index_step = math.floor(predict_time / TIME_PERIOD) + 1  # must be an integer
        # prints when too much data was skipped - some data will not be predicted at all
        if index_step > buffer_length:
            print(index_step, "data skipped is too high")
        # ensures the last sample is not missed
        if (i + index_step) >= len(motion) and i != (len(motion) - 1):
            i = len(motion) - 2  # -2 to counteract the effect of index_step = 1
            index_step = 1
        i += index_step

    print("Finished", len(total_predictions) + buffer_length, "/", len(motion), "predictions!")
    return total_predictions, predicting_times, sum(reading_times)


# evaluates model: Prints performance + accuracy and plots graphs
def evaluate_model(times, data, hyperparameters, start_index, total_predictions, TIME_PERIOD):
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
    max_index_skipped = np.floor(np.divide(max_predicting_times, TIME_PERIOD)) + 1
    total_prediction_time = [sum(predicting_times[0]), sum(predicting_times[1]), sum(predicting_times[2])]
    print(
        "\nTotal time filling buffer:", np.max(total_reading_time),
        "\nTotal time filtering buffer (generating labels):", np.max(total_filtering_time),
        "\nTotal time taken during training/tuning:", np.max(training_time),
        "\nMaximum time taken for a prediction [X, Y, Z]:", max_predicting_times,
        "\nAverage time taken for a prediction [X, Y, Z]:", avg_predicting_times,
        "\nMinimum time taken for a prediction [X, Y, Z]:", min_predicting_times,
        "\nMaximum samples per prediction loop [X, Y, Z]:", max_index_skipped,
        "\nTotal prediction time:",
        np.max(np.add(total_prediction_time, wait_time)), "/",
        (len(predicting_times[0]) * TIME_PERIOD)
    )

    # truncates the data to the same length as the predictions
    motion = [data[1][start_index:], data[2][start_index:], data[3][start_index:]]
    # percentage of data not predicted
    data_loss = [
        100 * (1 - (len(total_predictions[0]) / len(motion[0]))),  # X
        100 * (1 - (len(total_predictions[1]) / len(motion[1]))),  # Y
        100 * (1 - (len(total_predictions[2]) / len(motion[2])))  # Z
    ]
    print("\nData loss [X, Y, Z]:", data_loss)
    # outputs the hyperparameter values
    print("Hyperparameters:", hyperparameters)

    # interpolates the motion data to be the same length as the results and shortens the graph (better view)
    for i in range(len(total_predictions)):
        # fills the gaps in the predictions list caused by skipping samples during prediction
        interp_pred = interpolate.interp1d(np.arange(len(total_predictions[i])), total_predictions[i])
        stretched_pred = interp_pred(np.linspace(0, len(total_predictions[i]) - 1, len(motion[i])))
        total_predictions[i] = stretched_pred
        # selects the last 20% of data to show more detail in graph and also to remove bad data at the beginning
        total_predictions[i] = total_predictions[i][round(0.8 * len(total_predictions[i])):len(total_predictions[i])]
        motion[i] = motion[i][round(0.8 * len(motion[i])):len(motion[i])]

    filtered_motion = []
    accuracy = [[], []]  # [R2, NRMSE]
    # calculates the labels and accuracy of the truncated data
    for i in range(len(motion)):
        filtered_motion.append(dh.filter_data(motion[i], TIME_PERIOD))
        [temp_R2, temp_rmse] = eva.calc_accuracy(filtered_motion[i], total_predictions[i])
        accuracy[0].append(temp_R2)  # [X, Y, Z]
        accuracy[1].append(temp_rmse)  # [X, Y, Z]
    # prints the accuracies of the overall voluntary motion (after completion)
    print(
        "\nAccuracy",
        "\nX [R2, NRMSE]: [" + str(accuracy[0][0]) + "%" + ", " + str(accuracy[1][0]) + "]",
        "\nY [R2, NRMSE]: [" + str(accuracy[0][1]) + "%" + ", " + str(accuracy[1][1]) + "]",
        "\nZ [R2, NRMSE]: [" + str(accuracy[0][2]) + "%" + ", " + str(accuracy[1][2]) + "]"
    )

    # gets the tremor component by subtracting from the voluntary motion
    actual_tremor = []
    predicted_tremor = []
    tremor_accuracy = [[], []]  # [R2, NRMSE]
    for i in range(len(motion)):
        actual_tremor.append(np.subtract(motion[i], filtered_motion[i]))
        predicted_tremor.append(np.subtract(motion[i], total_predictions[i]))
        [temp_R2, temp_rmse] = eva.calc_tremor_accuracy(motion[i], total_predictions[i], filtered_motion[i])
        tremor_accuracy[0].append(temp_R2)
        tremor_accuracy[1].append(temp_rmse)
    tremor_error = np.subtract(actual_tremor, predicted_tremor)
    # prints the accuracies of the overall tremor component (after completion)
    print(
        "\nTremor accuracy",
        "\nX [R2, NRMSE]: [" + str(tremor_accuracy[0][0]) + "%" + ", " + str(tremor_accuracy[1][0]) + "]",
        "\nY [R2, NRMSE]: [" + str(tremor_accuracy[0][1]) + "%" + ", " + str(tremor_accuracy[1][1]) + "]",
        "\nZ [R2, NRMSE]: [" + str(tremor_accuracy[0][2]) + "%" + ", " + str(tremor_accuracy[1][2]) + "]"
    )

    # puts regression model data in a list
    model_data = [
        [motion[0], filtered_motion[0], total_predictions[0], "X motion (mm)"],
        [motion[1], filtered_motion[1], total_predictions[1], "Y motion (mm)"],
        [motion[2], filtered_motion[2], total_predictions[2], "Z motion (mm)"]
    ]
    model_axes_labels = ["Original signal", "Zero phase filter", "Prediction"]
    model_data_title = "Graph showing voluntary motion of model"
    # puts the tremor component data in a list
    tremor_data = [
        [actual_tremor[0], predicted_tremor[0], tremor_error[0], "X motion (mm)"],
        [actual_tremor[1], predicted_tremor[1], tremor_error[1], "Y motion (mm)"],
        [actual_tremor[2], predicted_tremor[2], tremor_error[2], "Z motion (mm)"]
    ]
    tremor_axes_labels = ["Actual tremor", "Predicted tremor", "Tremor error"]
    tremor_data_title = "Graph showing tremor component of model"

    t = np.array(data[0], dtype='f') * TIME_PERIOD  # samples are measured at a rate of 250Hz
    pltr.plot_model(t[len(t) - len(total_predictions[0]):], model_data, model_axes_labels, model_data_title)  # plots SVR model
    pltr.plot_model(t[len(t) - len(total_predictions[0]):], tremor_data, tremor_axes_labels, tremor_data_title)  # plots the tremor components

    return hyperparameters, accuracy, tremor_accuracy, np.max(training_time), avg_predicting_times


if __name__ == '__main__':
    model = "SVM"
    # model = "Random Forest"

    start_predictor("./data/real_tremor_data.csv", model)
