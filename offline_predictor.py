# libraries imported
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


def main(FILE_NAME, MODEL_TYPE):
    training_testing_ratio = 0.6
    TIME_PERIOD = 1 / 250

    start_time = datetime.now()
    # reads data into memory and filters it
    data = dh.read_data(FILE_NAME, 200, 5000)  # real tremor data (t, x, y, z, grip force)
    filtered_data = filter_data(data)  # filters data to get an estimate of intended movement (label)
    # 80% of data is used for training
    training_data = data[:, :int(training_testing_ratio * len(data[0]))]  # first 80% of data for training
    filtered_training_data = filtered_data[:, :int(training_testing_ratio * len(filtered_data[0]))]  # training labels
    time = np.array(data[0], dtype='f') * TIME_PERIOD  # samples are measured at a rate of 250Hz
    end_time = datetime.now()
    # time taken to read and split the data
    data_reading_time = (end_time - start_time).total_seconds()

    start_time = datetime.now()

    # training data is assigned
    training_motion = [training_data[1], training_data[2], training_data[3]]  # [X, Y, Z]
    training_label = [filtered_training_data[1], filtered_training_data[2], filtered_training_data[3]]  # [X, Y, Z]
    # training data and labels are normalised
    for i in range(len(training_motion)):
        training_motion[i] = fh.normalise(training_motion[i])
        training_label[i] = fh.normalise(training_label[i])

    end_time = datetime.now()
    # time taken to select and normalise useful training data
    selecting_training_time = (end_time - start_time).total_seconds()

    start_time = datetime.now()

    # calculates the features in a separate function
    [training_features, horizon] = fh.gen_all_features(training_motion, training_label)
    # prints the features and the average horizon
    print("\nTraining features (x, y, z):\n" + str(np.array(training_features)))
    print("Horizon values [x, y, z]:", horizon)

    end_time = datetime.now()
    # time taken to create training features
    training_features_time = (end_time - start_time).total_seconds()

    # SVM with rbf kernel
    regression = []
    hyperparameters = []
    tuned_training_time = []  # time taken to tune and train model (for each axis)
    print("Tuning...")
    for i in range(len(training_features)):
        start_time = datetime.now()

        # reformats the features for fitting the model (numpy array)
        axis_features = np.vstack(training_features[i]).T
        # tunes and trains the regression model
        [temp_reg, temp_params] = op.tune_model(axis_features, training_label[i], MODEL_TYPE)
        regression.append(temp_reg)
        hyperparameters.append(temp_params)

        end_time = datetime.now()
        tuned_training_time.append((end_time - start_time).total_seconds())
    print("Done!")
    print("\nHyperparameters (x, y, z):\n" + str(hyperparameters))

    start_time = datetime.now()

    # 20% of the data is separated and used for testing
    test_data = data[:, int(training_testing_ratio * len(data[0])):]  # last 20% of data for testing
    filtered_test_data = filtered_data[:, int(training_testing_ratio * len(filtered_data[0])):]  # testing labels
    # test data is assigned
    test_motion = [test_data[1], test_data[2], test_data[3]]  # [X, Y, Z]
    test_label = [filtered_test_data[1], filtered_test_data[2], filtered_test_data[3]]  # [X, Y, Z]
    norm_motion = []
    norm_label = []
    # test data is normalised but the original values are kept
    for i in range(len(test_motion)):
        norm_motion.append(fh.normalise(test_motion[i]))
        norm_label.append(fh.normalise(test_label[i]))

    end_time = datetime.now()
    # time taken to select and normalise useful test data
    selecting_test_time = (end_time - start_time).total_seconds()

    start_time = datetime.now()

    # calculates the features in a separate function
    test_features = fh.gen_all_features(norm_motion, norm_label, horizon)
    print("\nTest features (x):\n", np.array(test_features[0]))
    print("Test features (y):\n", np.array(test_features[1]))
    print("Test features (z):\n", np.array(test_features[2]))

    end_time = datetime.now()
    # time taken to create test data features
    test_features_time = (end_time - start_time).total_seconds()

    # predicts intended motion using the original data as an input (scaled to intended motion)
    prediction = []
    predicting_time = []  # time taken to predict voluntary motion (for each axis)
    for i in range(len(test_features)):
        start_time = datetime.now()

        axis_features = np.vstack(test_features[i]).T  # reformats the features for fitting the model (numpy array)
        # rescales the output to match the actual value
        prediction.append(fh.match_scale(test_label[i], regression[i].predict(axis_features)))

        end_time = datetime.now()
        predicting_time.append((end_time - start_time).total_seconds())
    print("\nPredicted output (x):\n", np.array(prediction[0]), "\nActual output (x):\n", np.array(test_label[0]))
    print("\nPredicted output (y):\n", np.array(prediction[1]), "\nActual output (y):\n", np.array(test_label[1]))
    print("\nPredicted output (z):\n", np.array(prediction[2]), "\nActual output (z):\n", np.array(test_label[2]))

    # calculates and prints the R2 score and normalised RMSE of the model
    accuracy = []
    for i in range(len(test_label)):
        accuracy.append(eva.calc_accuracy(test_label[i], prediction[i]))
    print("\nX Accuracy [R2, NRMSE]: " + "[" + str(accuracy[0][0]) + "%" + ", " + str(accuracy[0][1]) + "]")
    print("Y Accuracy [R2, NRMSE]: " + "[" + str(accuracy[1][0]) + "%" + ", " + str(accuracy[1][1]) + "]")
    print("Z Accuracy [R2, NRMSE]: " + "[" + str(accuracy[2][0]) + "%" + ", " + str(accuracy[2][1]) + "]")

    # gets the tremor component by subtracting from the voluntary motion
    actual_tremor = []
    predicted_tremor = []
    tremor_accuracy = []
    for i in range(len(test_motion)):
        actual_tremor.append(np.subtract(test_motion[i], test_label[i]))
        predicted_tremor.append(np.subtract(test_motion[i], prediction[i]))
        # calculates the normalised RMSE of the tremor component
        tremor_accuracy.append(eva.calc_accuracy(actual_tremor[i], predicted_tremor[i]))
    # converts and prints a the NRMSE in a percentage form
    print("X Tremor accuracy [R2, NRMSE]: " +
          "[" + str(tremor_accuracy[0][0]) + "%" + ", " + str(tremor_accuracy[0][1]) + "]")
    print("Y Tremor accuracy [R2, NRMSE]: " +
          "[" + str(tremor_accuracy[1][0]) + "%" + ", " + str(tremor_accuracy[1][1]) + "]")
    print("Z Tremor accuracy [R2, NRMSE]: " +
          "[" + str(tremor_accuracy[2][0]) + "%" + ", " + str(tremor_accuracy[2][1]) + "]")

    # shortens data list length to show more detail in graphs
    for i in range(len(test_motion)):
        test_motion[i] = test_motion[i][round(0.8 * len(test_motion[i])):]
        test_label[i] = test_label[i][round(0.8 * len(test_label[i])):]
        actual_tremor[i] = actual_tremor[i][round(0.8 * len(actual_tremor[i])):]
        predicted_tremor[i] = predicted_tremor[i][round(0.8 * len(predicted_tremor[i])):]
        prediction[i] = prediction[i][round(0.8 * len(prediction[i])):]
    tremor_error = np.subtract(actual_tremor, predicted_tremor)

    # puts regression model data in a list
    model_data = [
        [test_motion[0], test_label[0], prediction[0], "X motion (mm)"],
        [test_motion[1], test_label[1], prediction[1], "Y motion (mm)"],
        [test_motion[2], test_label[2], prediction[2], "Z motion (mm)"]
    ]
    model_axes_labels = ["Original signal", "Filtered output", "Predicted output"]
    model_data_title = "Graph showing voluntary motion of model"
    # puts the tremor component data in a list
    tremor_data = [
        [actual_tremor[0], predicted_tremor[0], tremor_error[0], "X motion (mm)"],
        [actual_tremor[1], predicted_tremor[1], tremor_error[1], "Y motion (mm)"],
        [actual_tremor[2], predicted_tremor[2], tremor_error[2], "Z motion (mm)"]
    ]
    tremor_axes_labels = ["Actual tremor", "Predicted tremor", "Tremor error"]
    tremor_data_title = "Graph showing tremor component of model"
    # puts all features in a list for passing to the plot function (feature | legend)
    features_data = [
        [
            [test_features[0][0], "Motion (x)"],
            [test_features[0][1], "Velocity (x)"],
            [test_features[0][2], "Acceleration (x)"],
            [test_features[0][3], "Past motion (x)"],
            [test_features[0][4], "Average motion (x)"]
        ],
        [
            [test_features[1][0], "Motion (y)"],
            [test_features[1][1], "Velocity (y)"],
            [test_features[1][2], "Acceleration (y)"],
            [test_features[1][3], "Past motion (y)"],
            [test_features[1][4], "Average motion (y)"]
        ],
        [
            [test_features[2][0], "Motion (z)"],
            [test_features[2][1], "Velocity (z)"],
            [test_features[2][2], "Acceleration (z)"],
            [test_features[2][3], "Past motion (z)"],
            [test_features[2][4], "Average motion (z)"]
        ]
    ]

    # time is shortened to match the length of the test data
    time = time[int(training_testing_ratio * len(data[0])):]
    plt.plot_model(time[round(0.8 * len(time)):], model_data, model_axes_labels, model_data_title)  # plots SVR model
    plt.plot_model(time[round(0.8 * len(time)):], tremor_data, tremor_axes_labels, tremor_data_title)  # plots the tremor components
    # plots the features
    for axis in features_data:
        plt.plot_data(time, axis, "Time (s)", "N-motion")

    # prints performance of the program
    print(
        "\nPerformance:\n==================================",
        "\nTime taken to read data:", data_reading_time,
        "\nTime taken to select and normalise data for creating training features:", selecting_training_time,
        "\nTime taken to generate features for training:", training_features_time,
        "\nTime taken to tune and train regression model:",
        "\n\tX axis =", tuned_training_time[0],
        "\n\tY axis =", tuned_training_time[1],
        "\n\tZ axis =", tuned_training_time[2],
        "\nTime taken to select and normalise data for creating training features:", selecting_test_time,
        "\nTime taken to generate features for testing/predicting:", test_features_time,
        "\nTime taken to predict voluntary motion:",
        "\n\tX axis =", predicting_time[0],
        "\n\tY axis =", predicting_time[1],
        "\n\tZ axis =", predicting_time[2],
        "\nTotal time taken:",
        (data_reading_time + selecting_training_time + training_features_time + tuned_training_time[0] +
         tuned_training_time[1] + tuned_training_time[2] + selecting_test_time + test_features_time +
         predicting_time[0] + predicting_time[1] + predicting_time[2])
    )


# filters the input data to estimate the intended movement
def filter_data(data):
    time_period = 1 / 250
    nyquist = 1 / (2 * time_period)
    cut_off = 5 / nyquist

    # zero phase filter is used to generate the labels (slow but very accurate)
    [b, a] = signal.butter(2, cut_off, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


if __name__ == '__main__':
    file_name = "data/real_tremor_data.csv"
    # model_type = "SVM"
    model_type = "Random Forest"
    main(file_name, model_type)
