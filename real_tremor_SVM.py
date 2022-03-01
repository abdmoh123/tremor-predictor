# libraries imported
import csv
import matplotlib.pyplot as plt
from scipy import signal

from shared_functions import *  # functions that apply to both simulated and real tremor


def main():
    file_name = "real_tremor_data.csv"
    horizon = 5  # amount of data to be temporarily stored for feature creation

    # reads data into memory and filters it
    data = read_data(file_name, 200, 4000)  # real tremor data (t, x, y, z, grip force)
    filtered_data = filter_data(data)  # filters data to get an estimate of intended movement (label)
    print("Original data:\n", data)
    print("Filtered data:\n", filtered_data)

    t = data[0]  # horizontal component (time)
    x_motion = data[1]  # tremor in x axis
    x_label = filtered_data[1]  # intended movement in x axis
    y_motion = data[2]  # tremor in y axis
    y_label = filtered_data[2]  # intended movement in y axis
    z_motion = data[3]  # tremor in z axis
    z_label = filtered_data[3]  # intended movement in z axis
    grip_force = data[4]  # grip force pushed on the device by the user

    # plots data (x axis)
    plot_model(t, x_motion, x_label)


# plots the real tremor data
def plot_model(time, tremor, filtered_tremor):
    # plots filtered (labels) and unfiltered data in graph
    plt.plot(time, tremor, label="Training: Noisy tremor")
    plt.plot(time, filtered_tremor, label="Training: Intended movement")

    # displays graph (including legend)
    plt.legend()
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
