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

    time = np.array(data[0], dtype='f') / 250  # samples are measured at a rate of 250Hz
    [x_motion, x_motion_mean, x_motion_sigma] = fh.normalise(data[1], True)  # tremor in x axis (feature 1)
    [x_label, x_label_mean, x_label_sigma] = fh.normalise(filtered_data[1], True)  # intended motion in x axis
    [y_motion, y_motion_mean, y_motion_sigma] = fh.normalise(data[2], True)  # tremor in y axis (feature 1)
    [y_label, y_label_mean, y_label_sigma] = fh.normalise(filtered_data[2], True)  # intended motion in y axis
    [z_motion, z_motion_mean, z_motion_sigma] = fh.normalise(data[3], True)  # tremor in z axis (feature 1)
    [z_label, z_label_mean, z_label_sigma] = fh.normalise(filtered_data[3], True)  # intended motion in z axis

    x_prev_motion = fh.shift(x_motion, 1)  # uses the previous data element as a feature (0 if none)
    y_prev_motion = fh.shift(y_motion, 1)  # uses the previous data element as a feature (0 if none)
    z_prev_motion = fh.shift(z_motion, 1)  # uses the previous data element as a feature (0 if none)

    # calculates the rate of change of 3D motion
    x_velocity = fh.normalise(fh.calc_delta(time, x_motion))  # (feature 2)
    y_velocity = fh.normalise(fh.calc_delta(time, y_motion))  # (feature 2)
    z_velocity = fh.normalise(fh.calc_delta(time, z_motion))  # (feature 2)

    x_features = [x_motion, x_velocity, x_prev_motion]
    y_features = [y_motion, y_velocity, y_prev_motion]
    z_features = [z_motion, z_velocity, z_prev_motion]

    # finds the optimum value for C (regularisation parameter)
    print("Optimising models...")
    # [horizon_x, C_x] = mf.optimise(x_features, x_label)  # only required to run once
    # [horizon_y, C_y] = mf.optimise(y_features, y_label)  # only required to run once
    # [horizon_z, C_z] = mf.optimise(z_features, z_label)  # only required to run once
    print("Done!")
    C_x = 0.81
    horizon_x = 13
    C_y = 0.81
    horizon_y = 21
    C_z = 21.87
    horizon_z = 3
    print("Regularisation parameter C(x):", C_x, "\nHorizon value (x):", horizon_x)
    print("Regularisation parameter C(y):", C_y, "\nHorizon value (y):", horizon_y)
    print("Regularisation parameter C(z):", C_z, "\nHorizon value (z):", horizon_z)

    # calculates the average 3D motion
    avg_x = fh.normalise(fh.calc_average(x_motion, horizon_x))  # (feature 3)
    x_features.append(avg_x)
    avg_y = fh.normalise(fh.calc_average(y_motion, horizon_y))  # (feature 3)
    y_features.append(avg_y)
    avg_z = fh.normalise(fh.calc_average(z_motion, horizon_z))  # (feature 3)
    z_features.append(avg_z)

    # combines the features into 1 array
    x_features = np.vstack(x_features).T
    y_features = np.vstack(y_features).T
    z_features = np.vstack(z_features).T
    print("\nX Features:\n", x_features)
    print("Y Features:\n", y_features)
    print("Z Features:\n", z_features)

    # SVM with rbf kernel (x axis)
    x_regression = svm.SVR(kernel="rbf", C=C_x)
    y_regression = svm.SVR(kernel="rbf", C=C_y)
    z_regression = svm.SVR(kernel="rbf", C=C_z)
    # uses the features the fit the regression model to the data
    x_regression.fit(x_features, x_label)
    y_regression.fit(y_features, y_label)
    z_regression.fit(z_features, z_label)
    # predicts intended motion using the original data as an input (scaled to intended motion)
    x_predictions = fh.denormalise(x_regression.predict(x_features), x_label_mean, x_label_sigma)
    y_predictions = fh.denormalise(y_regression.predict(y_features), y_label_mean, y_label_sigma)
    z_predictions = fh.denormalise(z_regression.predict(z_features), z_label_mean, z_label_sigma)
    print("\nPredicted output (x):\n", x_predictions, "\nActual output (x):\n", filtered_data[1])
    print("\nPredicted output (y):\n", y_predictions, "\nActual output (y):\n", filtered_data[2])
    print("\nPredicted output (z):\n", z_predictions, "\nActual output (z):\n", filtered_data[3])

    # calculates and prints the normalised RMSE of the model
    x_accuracy = mf.calc_accuracy(x_predictions, filtered_data[1])
    y_accuracy = mf.calc_accuracy(y_predictions, filtered_data[2])
    z_accuracy = mf.calc_accuracy(z_predictions, filtered_data[3])
    print("\nAccuracy (x): " + str(x_accuracy) + "%")
    print("Accuracy (y): " + str(y_accuracy) + "%")
    print("Accuracy (z): " + str(z_accuracy) + "%\n")

    # denormalises the data (to its original scale)
    x_motion = fh.denormalise(x_motion, x_motion_mean, x_motion_sigma)
    x_label = fh.denormalise(x_label, x_label_mean, x_label_sigma)
    y_motion = fh.denormalise(y_motion, y_motion_mean, y_motion_sigma)
    y_label = fh.denormalise(y_label, y_label_mean, y_label_sigma)
    z_motion = fh.denormalise(z_motion, z_motion_mean, z_motion_sigma)
    z_label = fh.denormalise(z_label, z_label_mean, z_label_sigma)

    # gets the tremor component by subtracting from the voluntary motion
    x_actual_tremor = np.subtract(x_motion, x_label)
    x_predicted_tremor = np.subtract(x_motion, x_predictions)
    x_tremor_error = np.subtract(x_actual_tremor, x_predicted_tremor)
    y_actual_tremor = np.subtract(y_motion, y_label)
    y_predicted_tremor = np.subtract(y_motion, y_predictions)
    y_tremor_error = np.subtract(y_actual_tremor, y_predicted_tremor)
    z_actual_tremor = np.subtract(z_motion, z_label)
    z_predicted_tremor = np.subtract(z_motion, z_predictions)
    z_tremor_error = np.subtract(z_actual_tremor, z_predicted_tremor)
    # calculates and prints the normalised RMSE percentage of the tremor component
    x_tremor_accuracy = mf.calc_accuracy(x_predicted_tremor, x_actual_tremor)
    y_tremor_accuracy = mf.calc_accuracy(y_predicted_tremor, y_actual_tremor)
    z_tremor_accuracy = mf.calc_accuracy(z_predicted_tremor, z_actual_tremor)
    print("Tremor accuracy (x): " + str(x_tremor_accuracy) + "%")
    print("Tremor accuracy (y): " + str(y_tremor_accuracy) + "%")
    print("Tremor accuracy (z): " + str(z_tremor_accuracy) + "%")

    # puts all features in a list for passing to the plot function
    x_features = [
        [fh.normalise(x_motion), "Motion (x)"],
        [avg_x, "Average motion (x)"],
        [x_prev_motion, "Previous motion (x)"],
        [x_velocity, "Velocity (x)"]
    ]
    y_features = [
        [fh.normalise(y_motion), "Motion (y)"],
        [avg_y, "Average motion (y)"],
        [y_prev_motion, "Previous motion (y)"],
        [y_velocity, "Velocity (y)"]
    ]
    z_features = [
        [fh.normalise(z_motion), "Motion (z)"],
        [avg_z, "Average motion (z)"],
        [z_prev_motion, "Previous motion (z)"],
        [z_velocity, "Velocity (z)"]
    ]
    # puts the tremor component data in lists
    x_tremors = [
        [x_actual_tremor, "Actual tremor (x)"],
        [x_predicted_tremor, "Predicted tremor (x)"],
        [x_tremor_error, "Tremor error (x)"]
    ]
    y_tremors = [
        [y_actual_tremor, "Actual tremor (y)"],
        [y_predicted_tremor, "Predicted tremor (y)"],
        [y_tremor_error, "Tremor error (y)"]
    ]
    z_tremors = [
        [z_actual_tremor, "Actual tremor (z)"],
        [z_predicted_tremor, "Predicted tremor (z)"],
        [z_tremor_error, "Tremor error (z)"]
    ]

    plot_model(time, x_motion, x_label, x_predictions, "X motion (mm)")  # plots SVR model (x axis)
    plot_model(time, y_motion, y_label, y_predictions, "Y motion (mm)")  # plots SVR model (y axis)
    plot_model(time, z_motion, z_label, z_predictions, "Z motion (mm)")  # plots SVR model (z axis)
    plot_data(time, x_tremors, "X motion (mm)")  # plots the tremor components (x axis) in a separate graph
    plot_data(time, y_tremors, "Y motion (mm)")  # plots the tremor components (y axis) in a separate graph
    plot_data(time, z_tremors, "Z motion (mm)")  # plots the tremor components (z axis) in a separate graph
    plot_data(time, x_features, "X motion (n)")  # plots the features (x axis) in a separate graph
    plot_data(time, y_features, "Y motion (n)")  # plots the features (y axis) in a separate graph
    plot_data(time, z_features, "Z motion (n)")  # plots the features (z axis) in a separate graph


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
