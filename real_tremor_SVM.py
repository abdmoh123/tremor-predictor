# libraries imported
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn import svm
from sklearn.preprocessing import normalize

# functions that apply to both simulated and real tremor
import functions.feature_handler as fh
import functions.miscellaneous as mf
import functions.evaluator as eva
import functions.optimiser as op

np.set_printoptions(threshold=50)  # shortens long arrays in the console window


def main():
    file_name = "./data/real_tremor_data.csv"
    training_testing_ratio = 0.8

    # reads data into memory and filters it
    data = read_data(file_name, 200, 4000)  # real tremor data (t, x, y, z, grip force)
    filtered_data = filter_data(data)  # filters data to get an estimate of intended movement (label)
    # 80% of data is used for training
    training_data = data[:, :int(training_testing_ratio * len(data[0]))]  # first 80% of data for training
    filtered_training_data = filtered_data[:, :int(training_testing_ratio * len(filtered_data[0]))]  # training labels
    time = np.array(data[0], dtype='f') / 250  # samples are measured at a rate of 250Hz

    # training data is assigned
    [x, y, z] = select_normalised_data(training_data)  # normalises and assigns the input motion data
    [fx, fy, fz] = select_normalised_data(filtered_training_data)  # normalises and assigns the voluntary motion data
    # motion and labels [0] = X axis, [1] = Y axis, [2] = Z axis
    training_motion = [x[0], y[0], z[0]]
    training_label = [fx[0], fy[0], fz[0]]

    # calculates the features in a separate function
    [training_features, horizon, C] = prepare_model(time, training_motion, training_label)

    # SVM with rbf kernel (x axis)
    regression = []
    for i in range(len(training_features)):
        # reformats the features for fitting the model (numpy array)
        axis_features = np.vstack(training_features[i]).T
        regression.append(svm.SVR(kernel="rbf", C=C[i]))
        # uses the features the fit the regression model to the data
        regression[i].fit(axis_features, training_label[i])
    print("\nTraining features (x):\n", np.array(training_features[0]))
    print("Training features (y):\n", np.array(training_features[1]))
    print("Training features (z):\n", np.array(training_features[2]))

    # 20% of the data is separated and used for testing
    test_data = data[:, int(training_testing_ratio * len(data[0])):]  # last 20% of data for testing
    filtered_test_data = filtered_data[:, int(training_testing_ratio * len(filtered_data[0])):]  # testing labels
    # test data is assigned
    [xt, yt, zt] = select_normalised_data(test_data)  # normalises and assigns the input motion data
    [fxt, fyt, fzt] = select_normalised_data(filtered_test_data)  # normalises and assigns the voluntary motion data

    # motion and labels [0] = X axis, [1] = Y axis, [2] = Z axis
    test_motion = [xt[0], yt[0], zt[0]]
    test_motion_mean = [xt[1], yt[1], zt[1]]
    test_motion_sigma = [xt[2], yt[2], zt[2]]
    test_label = [fxt[0], fyt[0], fzt[0]]
    test_label_mean = [fxt[1], fyt[1], fzt[1]]
    test_label_sigma = [fxt[2], fyt[2], fzt[2]]

    # calculates the features in a separate function
    test_features = prepare_model(time, test_motion, test_label, horizon)
    # predicts intended motion using the original data as an input (scaled to intended motion)
    prediction = []
    for i in range(len(test_features)):
        axis_features = np.vstack(test_features[i]).T   # reformats the features for fitting the model (numpy array)
        prediction.append(fh.denormalise(regression[i].predict(axis_features), test_label_mean[i], test_label_sigma[i]))
    print("\nTest features (x):\n", np.array(test_features[0]))
    print("Test features (y):\n", np.array(test_features[1]))
    print("Test features (z):\n", np.array(test_features[2]))

    # denormalises the data + labels (to its original scale)
    motion = []
    label = []
    for i in range(len(test_motion)):
        motion.append(fh.denormalise(test_motion[i], test_motion_mean[i], test_motion_sigma[i]))
        label.append(fh.denormalise(test_label[i], test_label_mean[i], test_label_sigma[i]))
    print("\nPredicted output (x):\n", np.array(prediction[0]), "\nActual output (x):\n", np.array(label[0]))
    print("\nPredicted output (y):\n", np.array(prediction[1]), "\nActual output (y):\n", np.array(label[1]))
    print("\nPredicted output (z):\n", np.array(prediction[2]), "\nActual output (z):\n", np.array(label[2]))

    # calculates and prints the normalised RMSE of the model
    accuracy = []
    for i in range(len(label)):
        accuracy.append(eva.calc_accuracy(label[i], prediction[i]))
    print("\nAccuracy (x): " + str(100 * (1 - accuracy[0])) + "%")
    print("Accuracy (y): " + str(100 * (1 - accuracy[1])) + "%")
    print("Accuracy (z): " + str(100 * (1 - accuracy[2])) + "%\n")

    # gets the tremor component by subtracting from the voluntary motion
    actual_tremor = []
    predicted_tremor = []
    tremor_accuracy = []
    for i in range(len(motion)):
        actual_tremor.append(np.subtract(motion[i], label[i]))
        predicted_tremor.append(np.subtract(motion[i], prediction[i]))
        # calculates the normalised RMSE of the tremor component
        tremor_accuracy.append(eva.calc_accuracy(actual_tremor[i], predicted_tremor[i]))
    tremor_error = np.subtract(actual_tremor, predicted_tremor)
    # converts and prints a the NRMSE in a percentage form
    print("Tremor accuracy (x): " + str(100 * (1 - tremor_accuracy[0])) + "%")
    print("Tremor accuracy (y): " + str(100 * (1 - tremor_accuracy[1])) + "%")
    print("Tremor accuracy (z): " + str(100 * (1 - tremor_accuracy[2])) + "%")

    # puts all features in a list for passing to the plot function (feature | legend)
    plot_features = [
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
    # puts the tremor component data in lists (tremor | legend)
    plot_tremors = [
        [
            [actual_tremor[0], "Actual tremor (x)"],
            [predicted_tremor[0], "Predicted tremor (x)"],
            [tremor_error[0], "Tremor error (x)"]
        ],
        [
            [actual_tremor[1], "Actual tremor (y)"],
            [predicted_tremor[1], "Predicted tremor (y)"],
            [tremor_error[1], "Tremor error (y)"]
        ],
        [
            [actual_tremor[2], "Actual tremor (z)"],
            [predicted_tremor[2], "Predicted tremor (z)"],
            [tremor_error[2], "Tremor error (z)"]
        ]
    ]
    # time is shortened to match the length of the test data
    time = time[int(training_testing_ratio * len(data[0])):]
    # plots SVR model
    plot_model(time, motion[0], label[0], prediction[0], "X motion (mm)")  # x axis
    plot_model(time, motion[1], label[1], prediction[1], "Y motion (mm)")  # y axis
    plot_model(time, motion[2], label[2], prediction[2], "Z motion (mm)")  # z axis
    # plots the tremor components
    for axis in plot_tremors:
        plot_data(time, axis, "Motion (mm)")
    # plots the features
    for axis in plot_features:
        plot_data(time, axis, "N-motion")


def prepare_model(time, motion, labels, horizon=None):
    velocity = []  # feature 2
    acceleration = []  # feature 3
    past_motion = []  # feature 4
    for i in range(len(motion)):
        # calculates the rate of change of 3D motion
        velocity.append(fh.calc_delta(time, motion[i]))
        # calculates the rate of change of rate of change of 3D motion (rate of change of velocity)
        acceleration.append(fh.calc_delta(time, velocity[i]))
        # uses the past data as a feature
        past_motion.append(fh.normalise(fh.shift(motion[i])))  # previous value

        # smoothing the velocity and acceleration
        velocity[i] = fh.normalise(fh.calc_average(velocity[i], 5))
        acceleration[i] = fh.normalise(fh.calc_average(acceleration[i], 5))

    # finds the optimum C and horizon values if no horizon values are inputted
    if horizon is None:
        features = []
        # puts all existing features in a list for model optimisation
        for i in range(len(motion)):
            features.append([
                motion[i],
                velocity[i],
                acceleration[i],
                past_motion[i]
            ])

        # generates optimal horizon and C values
        [horizon, C] = optimise_model(features, labels)

        for i in range(len(motion)):
            # calculates the average 3D motion
            average = fh.normalise(fh.calc_average(motion[i], horizon[i]))  # last feature
            # adds the average feature to the features list
            features[i].append(average)
        return features, horizon, C
    else:
        features = []
        # puts existing features in a list
        for i in range(len(motion)):
            features.append([
                motion[i],
                velocity[i],
                acceleration[i],
                past_motion[i]
            ])

        for i in range(len(motion)):
            # calculates the average 3D motion
            average = fh.normalise(fh.calc_average(motion[i], horizon[i]))  # last feature
            # adds the average feature to the features list
            features[i].append(average)
        return features


def optimise_model(features, labels):
    # finds the optimum value for C (regularisation parameter)
    # print("Optimising models...")
    # C = []
    # horizon = []
    # # only required to run once
    # for i in range(len(features)):
    #     C.append(op.optimise_c(features[i], labels[i]))
    #     horizon.append(op.optimise_parameter(features[i], labels[i], "horizon"))
    # print("Done!")

    # used to save time (optimising is only required once)
    C = [0.81, 0.81, 0.81]  # X, Y, Z
    horizon = [11, 27, 35]  # X, Y, Z

    # prints the optimised values
    print("Regularisation parameter C(x):", C[0], "\nHorizon value (x):", horizon[0])
    print("Regularisation parameter C(y):", C[1], "\nHorizon value (y):", horizon[1])
    print("Regularisation parameter C(z):", C[2], "\nHorizon value (z):", horizon[2])
    return horizon, C


def select_normalised_data(data):
    x = fh.normalise(data[1], True)  # x axis (feature 1)
    y = fh.normalise(data[2], True)  # y axis (feature 1)
    z = fh.normalise(data[3], True)  # z axis (feature 1)
    return x, y, z


# plots the real tremor data and SVM model (x axis)
def plot_model(time, input_motion, label, predictions, y_axis_label):
    # splits plot window into 2 graphs
    fig, axes = plt.subplots(2)

    # plots data
    axes[0].plot(time, input_motion, label="Noisy data with tremor")
    axes[0].plot(time, label, label="Intended movement without tremor")
    axes[0].set(ylabel=y_axis_label)
    axes[0].legend()

    # plots SVM regression model
    axes[1].plot(time, predictions, label="SVM regression model")
    axes[1].plot(time, label, label="Intended movement without tremor")
    axes[1].set(ylabel=y_axis_label)
    axes[1].set(xlabel="Time (s)")
    axes[1].legend()

    # displays graphs
    plt.show()


# plots the tremor component of the data and predictions
def plot_data(time, data, y_axis_label):
    fig, axes = plt.subplots(len(data))

    # plots tremor data
    for i in range(len(data)):
        axes[i].plot(time, data[i][0], label=data[i][1], linewidth=0.5)
        axes[i].set(ylabel=y_axis_label)
        axes[i].legend()

    # axes labels and legend
    axes[len(data) - 1].set(xlabel="Time (s)")

    # displays graphs
    plt.show()


# filters the input data to estimate the intended movement
def filter_data(data):
    time_period = 1 / 250
    nyquist = 1 / (2 * time_period)
    cut_off = 5 / nyquist

    # zero phase filter is used to generate the labels (slow but very accurate)
    [b, a] = signal.butter(2, cut_off, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


# reads data in a csv file and puts them in a 2D list
def read_data(file_name, l_bound, u_bound):
    data = []
    with open(file_name, "r") as file:
        reader = csv.reader(file)
        rows = list(reader)

        # ensures bounds are valid
        [l_bound, u_bound] = mf.check_bounds(l_bound, u_bound, rows)

        # reads through the file and puts the data in the respective lists above
        for i in range(l_bound, u_bound):
            row = rows[i]
            data.append(list(np.float_(row)))
    # reshapes the list into a 2D numpy array with each feature/label being its own sub-array
    return np.vstack(data).T


if __name__ == '__main__':
    main()
