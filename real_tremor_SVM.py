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

    # calculates the rate of change of 3D motion
    x_velocity = fh.normalise(fh.calc_delta(time, x_motion))  # (feature 2)

    x_features = [x_motion, x_velocity]

    # finds the optimum value for C (regularisation parameter)
    # [horizon, C_x] = mf.optimise(x_features, x_label)  # only required to run once
    C_x = 21.87  # optimal C value = 21.87
    horizon = 5  # optimal horizon value = 5
    print("Regularisation parameter C(x):", C_x, "\nHorizon value:", horizon)

    # calculates the average 3D motion
    avg_x = fh.normalise(fh.calc_average(x_motion, horizon))  # (feature 3)
    x_features.append(avg_x)

    # combines the features into 1 array
    x_features = np.vstack(x_features).T
    print("\nX Features:\n", x_features)

    # SVM with rbf kernel (x axis)
    regression = svm.SVR(kernel="rbf", C=C_x)
    regression.fit(x_features, x_label)
    # predicts intended motion using the original data as an input (scaled to intended motion)
    predictions = fh.denormalise(regression.predict(x_features), x_label_mean, x_label_sigma)
    print("\nPredicted output:\n", predictions, "\nActual output:\n", filtered_data[1])

    # calculates and prints the normalised RMSE of the model
    accuracy = mf.calc_accuracy(predictions, filtered_data[1])
    print("\nAccuracy: " + str(accuracy) + "%")
    # denormalises the data (to its original scale)
    x_motion = fh.denormalise(x_motion, x_motion_mean, x_motion_sigma)
    x_label = fh.denormalise(x_label, x_label_mean, x_label_sigma)

    # gets the tremor component by subtracting from the voluntary motion
    actual_tremor = np.subtract(x_motion, x_label)
    predicted_tremor = np.subtract(predictions, x_label)
    tremor_error = np.subtract(actual_tremor, predicted_tremor)
    # calculates and prints the normalised RMSE percentage of the tremor component
    tremor_accuracy = mf.calc_accuracy(predicted_tremor, actual_tremor)
    print("Tremor accuracy: " + str(tremor_accuracy) + "%")

    # puts all features in a list for passing to the plot function
    features = [
        [fh.normalise(x_motion), "Motion (x)"],
        [avg_x, "Average motion (x)"],
        [x_velocity, "Velocity (x)"],
    ]
    # puts tremor component data in a list
    tremors = [
        [actual_tremor, "Actual tremor (x)"],
        [predicted_tremor, "Predicted tremor (x)"],
        [tremor_error, "Tremor error (x)"]
    ]

    plot_model(time, x_motion, x_label, predictions)  # plots SVR model (x axis)
    plot_data(time, tremors, "X motion (mm)")  # plots the tremor components (x axis) in a separate graph
    plot_data(time, features, "X motion (n)")  # plots the features (x axis) in a separate graph


# plots the real tremor data and SVM model (x axis)
def plot_model(time, input_motion, label, predictions):
    # splits plot window into 2 graphs
    fig, axes = plt.subplots(2)

    # plots data
    axes[0].plot(time, input_motion, label="Noisy data with tremor")
    axes[0].plot(time, label, label="Intended movement without tremor")
    axes[0].set(ylabel="X motion (mm)")
    axes[0].legend()

    # plots SVM regression model
    axes[1].plot(time, predictions, label="SVM regression model")
    axes[1].plot(time, label, label="Intended movement without tremor")
    axes[1].set(ylabel="X motion (mm)")
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
