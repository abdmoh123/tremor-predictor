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

    t = data[0]  # horizontal component (time)
    [x_motion, x_motion_mean, x_motion_sigma] = fh.normalise(data[1], True)  # tremor in x axis (feature 1)
    [x_label, x_label_mean, x_label_sigma] = fh.normalise(filtered_data[1], True)  # intended motion in x axis
    # y_motion = fh.normalise(data[2])  # tremor in y axis
    # y_label = filtered_data[2]  # intended motion in y axis
    # z_motion = fh.normalise(data[3])  # tremor in z axis
    # z_label = filtered_data[3]  # intended motion in z axis
    # grip_force = data[4]  # grip force pushed on the device by the user

    # calculates the rate of change of 3D motion
    delta_x = fh.normalise(fh.calc_delta(t, x_motion))  # (feature 2)

    # finds the optimum value for C (regularisation parameter)
    # [horizon, C_x] = mf.optimise([x_motion, delta_x], x_label)  # only required to run once (to find optimal values)
    C_x = 21.87  # optimal C value = 21.87
    horizon = 5  # optimal horizon value = 5
    print("Regularisation parameter C(x):", C_x, "\nHorizon value:", horizon)

    # calculates the average 3D motion
    avg_x = fh.normalise(fh.calc_average(x_motion, horizon))  # (feature 3)

    # combines the features into 1 array
    x_features = np.vstack((x_motion, delta_x, avg_x)).T
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

    # puts all data in a dictionary for passing to the plot function
    data_dict = {
        'x_motion': x_motion,
        'delta_x': delta_x,
        'avg_x': avg_x,
        'x_label': x_label,
        'predictions': predictions
    }
    # plots data and model (x axis)
    plot_model(t, data_dict)
    # puts tremor component data in a dictionary
    tremor_dict = {
        'actual_tremor': actual_tremor,
        'predicted_tremor': predicted_tremor,
        'tremor_error': tremor_error
    }
    # plots the tremor components (x axis) in a separate graph
    plot_tremor(t, tremor_dict)


# plots the real tremor data and SVM model (x axis)
def plot_model(time, data):
    # splits plot window into 2 graphs
    fig, axes = plt.subplots(3)

    # plots data
    axes[0].plot(time, data['x_motion'], label="Noisy data with tremor")
    axes[0].plot(time, data['x_label'], label="Intended movement without tremor")
    axes[0].legend()

    # plots SVM regression model
    axes[1].plot(time, data['predictions'], label="SVM regression model")
    axes[1].plot(time, data['x_label'], label="Intended movement without tremor")
    axes[1].set(ylabel="X motion voltage (V)")
    axes[1].legend()

    # plots the features (normalised)
    axes[2].plot(time, fh.normalise(data['x_motion']), label="X")
    axes[2].plot(time, fh.normalise(data['avg_x']), label="Avg X")
    axes[2].plot(time, fh.normalise(data['delta_x']), label="Delta X")
    axes[2].set(xlabel="time/index")
    axes[2].legend()

    # displays graphs
    plt.show()


# plots the tremor component of the data and predictions
def plot_tremor(time, data):
    # plots tremor data
    plt.plot(time, data['actual_tremor'], label="Actual tremor", linewidth=0.5)
    plt.plot(time, data['predicted_tremor'], label="Predicted tremor", linewidth=0.5)
    plt.plot(time, data['tremor_error'], label="Tremor error", linewidth=0.5)
    # axes labels and legend
    plt.ylabel("X motion voltage (V)")
    plt.xlabel("time/index")
    plt.legend()
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
