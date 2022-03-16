# libraries imported
import csv
import matplotlib.pyplot as plt
from scipy import signal

# functions that apply to both simulated and real tremor
from functions.feature_handler import *
from functions.miscellaneous import *


def main():
    file_name = "./data/real_tremor_data.csv"
    horizon = 10  # amount of data to be temporarily stored for feature creation

    # reads data into memory and filters it
    data = read_data(file_name, 200, 4000)  # real tremor data (t, x, y, z, grip force)
    filtered_data = filter_data(data)  # filters data to get an estimate of intended movement (label)
    print("Original data:\n", data)
    print("Filtered data:\n", filtered_data)

    t = data[0]  # horizontal component (time)
    [x_motion, x_motion_mean, x_motion_sigma] = normalise(data[1], True)  # tremor in x axis
    [x_label, x_label_mean, x_label_sigma] = normalise(filtered_data[1], True)  # intended motion in x axis
    # y_motion = normalise(data[2])  # tremor in y axis
    # y_label = filtered_data[2]  # intended motion in y axis
    # z_motion = normalise(data[3])  # tremor in z axis
    # z_label = filtered_data[3]  # intended motion in z axis
    # grip_force = data[4]  # grip force pushed on the device by the user

    # calculates the rate of change of 3D motion
    delta_x = normalise(calc_delta(t, x_motion))

    # calculates the average 3D motion
    avg_x = normalise(calc_average(x_motion, horizon))

    # combines the features into 1 array
    x_features = np.vstack((x_motion, delta_x, avg_x)).T
    print("X Features:\n", x_features)

    # finds the optimum value for C (regularisation parameter)
    # C_x = optimise_reg(x_features, x_label)  # only required to run once (to find optimum value)
    C_x = 1  # optimum C value = 1
    print("Regularisation parameter C(x):", C_x)

    # SVM with rbf kernel (x axis)
    regression = svm.SVR(kernel="rbf", C=C_x)
    regression.fit(x_features, x_label)
    # predicts intended motion using the original data as an input
    predictions = regression.predict(x_features)
    print("Predicted output:\n", predictions, "\nActual output:\n", filtered_data[1])

    # calculates and prints the RMSE of the model
    accuracy = calc_accuracy(predictions, x_label)
    print("\nAccuracy: " + str(accuracy) + "%")

    # denormalises the data (to its original scale)
    x_motion = denormalise(x_motion, x_motion_mean, x_motion_sigma)
    x_label = denormalise(x_label, x_label_mean, x_label_sigma)
    predictions = denormalise(predictions, x_label_mean, x_label_sigma)  # scaled to the intended motion
    # puts all data in a dictionary for passing to the plot function
    data_list = {
        'x_motion': x_motion,
        'delta_x': delta_x,
        'avg_x': avg_x,
        'x_label': x_label,
        'predictions': predictions
    }
    # plots data and model (x axis)
    plot_model(t, data_list)


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
    axes[2].plot(time, normalise(data['x_motion']), label="X")
    axes[2].plot(time, normalise(data['avg_x']), label="Avg X")
    axes[2].plot(time, normalise(data['delta_x']), label="Delta X")
    axes[2].set(xlabel="time/index")
    axes[2].legend()

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
        [l_bound, u_bound] = check_bounds(l_bound, u_bound, rows)

        # reads through the file and puts the data in the respective lists above
        for i in range(l_bound, u_bound):
            row = rows[i]
            data.append(list(np.float_(row)))
    # reshapes the list into a 2D numpy array with each feature/label being its own sub-array
    return np.vstack(data).T


if __name__ == '__main__':
    main()
