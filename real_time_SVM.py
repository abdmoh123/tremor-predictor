# libraries imported
import numpy as np
from datetime import datetime


# functions that apply to both simulated and real tremor
import functions.feature_handler as fh
import functions.data_handler as dh
import functions.evaluator as eva
import functions.optimiser as op
import functions.plotter as plt
# buffer class to simulate a real data buffer
from classes.buffer import Buffer

np.set_printoptions(threshold=50)  # shortens long arrays in the console window


def main():
    """ Constants """
    FILE_NAME = "./data/real_tremor_data.csv"
    TIME_PERIOD = 1 / 250  # a sample is recorded every 0.004 seconds
    N_SAMPLES = 500  # more samples = more accuracy but slower speed

    # reads data into memory and filters it
    data = dh.read_data(FILE_NAME, 200, 3000)  # real tremor data (t, x, y, z, grip force)

    """ Buffer filling phase """
    [motion_buffer, label_buffer, reading_times, filtering_time] = fill_buffers(data, N_SAMPLES, TIME_PERIOD)

    """ Training and tuning phase """
    [regression, horizon, training_time] = train_model(motion_buffer, label_buffer, TIME_PERIOD)

    """ Prediction phase """
    # skips all the samples being 'streamed' while the model was trained
    prediction_start = N_SAMPLES + round(training_time / TIME_PERIOD)  # index must be an integer
    print("Predictions start at index:", prediction_start)
    BUFFER_LENGTH = 10
    # performs the predictions using the trained model
    [total_predictions, predicting_times, wait_time] = predict_outputs(
        data,
        regression,
        horizon,
        prediction_start,
        BUFFER_LENGTH,
        TIME_PERIOD
    )

    """ Evaluation phase """
    times = [reading_times, filtering_time, training_time, predicting_times, wait_time]
    start_index = prediction_start + BUFFER_LENGTH
    evaluate_model(times, data, start_index, total_predictions, TIME_PERIOD)


# fills all buffers with data (in the beginning)
def fill_buffers(data, N_SAMPLES, TIME_PERIOD, prediction=False):
    motion_buffer = Buffer([], [], [], N_SAMPLES)
    label_buffer = Buffer([], [], [], N_SAMPLES)

    print("\nFilling buffer...")
    reading_times = []
    for i in range(N_SAMPLES):
        start_time = datetime.now()

        print("\nProgress:\n", int(100 * (i + 1) / N_SAMPLES), "%")  # prints progress (for testing purposes)

        current_motion = [data[1][i], data[2][i], data[3][i]]
        # buffer is updated
        motion_buffer.add(current_motion)

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

        return motion_buffer, label_buffer, reading_times, filtering_time
    else:
        return motion_buffer, reading_times


# trains and tunes a regression model (SVM)
def train_model(motion_buffer, label_buffer, TIME_PERIOD):
    start_time = datetime.now()

    # calculates the features in a separate function
    [features, horizon] = fh.gen_features(TIME_PERIOD, motion_buffer.normalise(), label_buffer.normalise())

    # SVM with rbf kernel (x axis)
    regression = []
    hyperparameters = []
    print("Tuning...")
    for j in range(len(features)):
        # reformats the features for fitting the model (numpy array)
        axis_features = np.vstack(features[j]).T
        # tunes and trains the regression model
        regression.append(op.tune_model(axis_features, label_buffer.normalise()[j]))
        hyperparameters.append(regression[j].best_params_)
    print("Done!")
    print("\nHyperparameters [{x}, {y}, {z}]:", hyperparameters)
    print("Horizon values [x, y, z]:", horizon)  # prints the optimised values

    end_time = datetime.now()
    # measures time taken for training the model
    training_time = (end_time - start_time).total_seconds()

    return regression, horizon, training_time


# predicts outputs using an already trained regression model (SVM)
def predict_outputs(data, regression, horizon, prediction_start, buffer_length, TIME_PERIOD):
    total_predictions = [[], [], []]
    predicting_times = []

    # skips data past prediction_start (time spent training the model)
    data = [
        data[0][prediction_start:],  # index
        data[1][prediction_start:],  # X
        data[2][prediction_start:],  # Y
        data[3][prediction_start:],  # Z
        data[4][prediction_start:],  # grip force
    ]

    # fills buffer in prediction mode (no label generation)
    [motion_buffer, reading_times] = fill_buffers(data, buffer_length, TIME_PERIOD, True)
    label_buffer = Buffer([], [], [], buffer_length)

    i = buffer_length  # skips all data already added to the buffer
    index_step = 1  # no skipping in the beginning
    while i < len(data[0]):
        start_time = datetime.now()

        # loop allows missed data to be saved to buffer
        for j in range(index_step, 0, -1):
            current_motion = [data[1][i - j], data[2][i - j], data[3][i - j]]
            motion_buffer.add(current_motion)
        label_buffer.content = motion_buffer.filter(TIME_PERIOD)

        # generates features out of the data in the buffer
        features = fh.gen_features(TIME_PERIOD, motion_buffer.normalise(), horizon=horizon)
        # gets midpoints and spreads to denormalise the predictions
        [midpoints, sigmas] = label_buffer.get_data_attributes()

        # predicts intended motion using the original data as an input (scaled to intended motion)
        prediction = []
        for j in range(len(features)):
            # reformats the features for fitting the model (numpy array)
            axis_features = np.vstack(features[j]).T
            # predicts the voluntary motion and denormalises it to the correct scale
            prediction.append(fh.denormalise(regression[j].predict(axis_features), midpoints[j], sigmas[j]))
            # selects and saves only the new predictions to an external array for evaluation
            new_predictions = prediction[j][len(prediction[j]) - index_step:len(prediction[j])]
            for value in new_predictions:
                total_predictions[j].append(value)

        # prints progress (for testing purposes)
        print("\nProgress:\n", int(100 * (i + prediction_start + 1) / (len(data[0]) + prediction_start)), "%")

        end_time = datetime.now()
        # measures time taken for predicting
        predicting_times.append((end_time - start_time).total_seconds())

        # ensures the last sample is not missed
        if (i + index_step) > len(data[0]) and i != (len(data[0]) - 1):
            index_step = len(data[0]) - i - 1
        else:
            # skips all the samples being 'streamed' while the program performed predictions
            index_step = round(predicting_times[len(predicting_times) - 1] / TIME_PERIOD)  # index must be an integer
        print("\nCurrent index:", i, "/", len(data[0]), ", Next index:", int(i + index_step))
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

    total_reading_time = sum(reading_times)
    total_filtering_time = filtering_time
    print(
        "\nTotal time filling buffer:", total_reading_time,
        "\nTotal time filtering data:", total_filtering_time,
        "\nTotal time taken during training/tuning:", training_time,
        "\nAverage time taken to predict voluntary motion:", np.mean(predicting_times),
        "\nAverage samples skipped during prediction:", round(np.mean(predicting_times) / TIME_PERIOD),
        "\nMaximum time taken for a prediction:", np.max(predicting_times),
        "\nMinimum time taken for a prediction:", np.min(predicting_times),
        "\nMaximum samples skipped during prediction:", round(np.max(predicting_times) / TIME_PERIOD),
        "\nTotal prediction time:", sum(predicting_times), "+", wait_time, "=", (sum(predicting_times) + wait_time)
    )

    # truncates the data to the same length as the predictions
    motion = [
        data[1][start_index:],
        data[2][start_index:],
        data[3][start_index:]
    ]

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

    actual_tremor = []
    predicted_tremor = []
    overall_tremor_accuracy = []
    # gets the tremor component by subtracting from the voluntary motion
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
    main()
