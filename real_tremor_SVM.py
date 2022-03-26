# libraries imported
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn import svm

# functions that apply to both simulated and real tremor
import functions.feature_handler as fh
import functions.miscellaneous as mf


def main():
    file_name = "./data/real_tremor_data.csv"

    # reads data into memory and filters it
    data = read_data(file_name, 200, 4000)  # real tremor data (t, x, y, z, grip force)
    filtered_data = filter_data(data)  # filters data to get an estimate of intended movement (label)
    training_data = data[:int(0.8 * len(data))]  # selects the first 80% of data for training
    filtered_training_data = filtered_data[:int(0.8 * len(filtered_data))]  # selects the labels for training
    time = np.array(data[0], dtype='f') / 250  # samples are measured at a rate of 250Hz

    [x, y, z] = select_normalised_data(training_data)  # normalises and assigns the input motion data
    [fx, fy, fz] = select_normalised_data(filtered_training_data)  # normalises and assigns the voluntary motion data

    # motion and labels [0] = X axis, [1] = Y axis, [2] = Z axis
    motion = [x[0], y[0], z[0]]
    motion_mean = [x[1], y[1], z[1]]
    motion_sigma = [x[2], y[2], z[2]]
    training_label = [fx[0], fy[0], fz[0]]
    label_mean = [fx[1], fy[1], fz[1]]
    label_sigma = [fx[2], fy[2], fz[2]]

    # calculates the features in a separate function
    [features, C] = prepare_model(time, motion, training_label)

    # reformats the features for fitting the model
    x_features = np.vstack(features[0]).T
    y_features = np.vstack(features[1]).T
    z_features = np.vstack(features[2]).T
    print("\nX Features:\n", x_features)
    print("Y Features:\n", y_features)
    print("Z Features:\n", z_features)

    # SVM with rbf kernel (x axis)
    x_regression = svm.SVR(kernel="rbf", C=C[0])
    y_regression = svm.SVR(kernel="rbf", C=C[1])
    z_regression = svm.SVR(kernel="rbf", C=C[2])
    # uses the features the fit the regression model to the data
    x_regression.fit(x_features, training_label[0])
    y_regression.fit(y_features, training_label[1])
    z_regression.fit(z_features, training_label[2])

    test_data = data[int(0.8 * len(data)):]  # selects the last 20% of data for testing
    filtered_test_data = filtered_data[int(0.8 * len(filtered_data)):]  # selects the labels for testing

    # predicts intended motion using the original data as an input (scaled to intended motion)
    predictions = [
        fh.denormalise(x_regression.predict(x_features), label_mean[0], label_sigma[0]),
        fh.denormalise(y_regression.predict(y_features), label_mean[1], label_sigma[1]),
        fh.denormalise(z_regression.predict(z_features), label_mean[2], label_sigma[2])
    ]
    print("\nPredicted output (x):\n", predictions[0], "\nActual output (x):\n", filtered_training_data[1])
    print("\nPredicted output (y):\n", predictions[1], "\nActual output (y):\n", filtered_training_data[2])
    print("\nPredicted output (z):\n", predictions[2], "\nActual output (z):\n", filtered_training_data[3])

    # calculates and prints the normalised RMSE of the model
    accuracy = [
        mf.calc_accuracy(predictions[0], filtered_training_data[1]),
        mf.calc_accuracy(predictions[1], filtered_training_data[2]),
        mf.calc_accuracy(predictions[2], filtered_training_data[3])
    ]
    print("\nAccuracy (x): " + str(100 * (1 - accuracy[0])) + "%")
    print("Accuracy (y): " + str(100 * (1 - accuracy[1])) + "%")
    print("Accuracy (z): " + str(100 * (1 - accuracy[2])) + "%\n")

    # denormalises the data (to its original scale)
    motion = [
        fh.denormalise(motion[0], motion_mean[0], motion_sigma[0]),  # X
        fh.denormalise(motion[1], motion_mean[1], motion_sigma[1]),  # Y
        fh.denormalise(motion[2], motion_mean[2], motion_sigma[2])  # Z
    ]
    label = [
        fh.denormalise(training_label[0], label_mean[0], label_sigma[0]),  # X
        fh.denormalise(training_label[1], label_mean[1], label_sigma[1]),  # Y
        fh.denormalise(training_label[2], label_mean[2], label_sigma[2])  # Z
    ]

    # gets the tremor component by subtracting from the voluntary motion
    actual_tremor = [
        np.subtract(motion[0], label[0]),  # X
        np.subtract(motion[1], label[1]),  # Y
        np.subtract(motion[2], label[2])  # Z
    ]
    predicted_tremor = [
        np.subtract(motion[0], predictions[0]),  # X
        np.subtract(motion[1], predictions[1]),  # Y
        np.subtract(motion[2], predictions[2])  # Z
    ]
    tremor_error = np.subtract(actual_tremor, predicted_tremor)
    # calculates and prints the normalised RMSE percentage of the tremor component
    tremor_accuracy = [
        mf.calc_accuracy(predicted_tremor[0], actual_tremor[0]),  # X
        mf.calc_accuracy(predicted_tremor[1], actual_tremor[1]),  # Y
        mf.calc_accuracy(predicted_tremor[2], actual_tremor[2])  # Z
    ]
    print("Tremor accuracy (x): " + str(100 * (1 - tremor_accuracy[0])) + "%")
    print("Tremor accuracy (y): " + str(100 * (1 - tremor_accuracy[1])) + "%")
    print("Tremor accuracy (z): " + str(100 * (1 - tremor_accuracy[2])) + "%")

    # puts all features in a list for passing to the plot function (feature | legend)
    plot_features = [
        [
            [features[0][0], "Motion (x)"],
            [features[0][1], "Velocity (x)"],
            [features[0][2], "Past motion (x)"],
            [features[0][3], "Average motion (x)"]
        ],
        [
            [features[1][0], "Motion (y)"],
            [features[1][1], "Velocity (y)"],
            [features[1][2], "Past motion (y)"],
            [features[1][3], "Average motion (y)"]
        ],
        [
            [features[2][0], "Motion (z)"],
            [features[2][1], "Velocity (z)"],
            [features[2][2], "Past motion (z)"],
            [features[2][3], "Average motion (z)"]
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
    # plots SVR model
    plot_model(time, motion[0], label[0], predictions[0], "X motion (mm)")  # x axis
    plot_model(time, motion[1], label[1], predictions[1], "Y motion (mm)")  # y axis
    plot_model(time, motion[2], label[2], predictions[2], "Z motion (mm)")  # z axis
    # plots the tremor components
    for axis in plot_tremors:
        plot_data(time, axis, "Motion (mm)")
    # plots the features
    for axis in plot_features:
        plot_data(time, axis, "N-motion")


def prepare_model(time, motion, labels):
    # calculates the rate of change of 3D motion
    velocity = [  # (feature 2)
        fh.normalise(fh.calc_delta(time, motion[0])),
        fh.normalise(fh.calc_delta(time, motion[1])),
        fh.normalise(fh.calc_delta(time, motion[2]))
    ]

    # uses the past data as a feature
    past_motion = [  # feature 3
        fh.shift(motion[0], 1),
        fh.shift(motion[1], 1),
        fh.shift(motion[2], 1)
    ]

    features = [
        [motion[0], velocity[0], past_motion[0]],
        [motion[1], velocity[1], past_motion[1]],
        [motion[2], velocity[2], past_motion[2]]
    ]

    # finds the optimum value for C (regularisation parameter)
    # print("Optimising models...")
    # [horizon_x, C_x] = mf.optimise(features[0], labels[0])  # only required to run once
    # [horizon_y, C_y] = mf.optimise(features[1], labels[1])  # only required to run once
    # [horizon_z, C_z] = mf.optimise(features[2], labels[2])  # only required to run once
    # print("Done!")
    C_x = 0.81
    horizon_x = 15
    C_y = 0.81
    horizon_y = 49
    C_z = 2.43
    horizon_z = 1
    print("Regularisation parameter C(x):", C_x, "\nHorizon value (x):", horizon_x)
    print("Regularisation parameter C(y):", C_y, "\nHorizon value (y):", horizon_y)
    print("Regularisation parameter C(z):", C_z, "\nHorizon value (z):", horizon_z)
    C = [C_x, C_y, C_z]

    # calculates the average 3D motion
    average = [  # (feature 4)
        fh.normalise(fh.calc_average(motion[0], horizon_x)),
        fh.normalise(fh.calc_average(motion[1], horizon_y)),
        fh.normalise(fh.calc_average(motion[2], horizon_z))
    ]
    # adds the average feature to the features list
    for i in range(len(features)):
        features[i].append(average[i])
    return features, C


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
