import csv
import numpy as np
from scipy import signal


# validates boundary values of an array/list (prevents errors)
def check_bounds(l_bound, u_bound, rows):
    # ensures that the lower bound is smaller than the upper bound
    if l_bound >= u_bound:
        l_bound = u_bound - 1
    # checks if bounds are valid (prevents index out of bounds error)
    if (l_bound < 0) | (l_bound >= len(rows)):
        l_bound = 0
    if (u_bound > len(rows)) | (u_bound <= 0):
        u_bound = len(rows)
    return l_bound, u_bound


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


# filters the input data to estimate the intended movement
def filter_data(data, TIME_PERIOD, zero_phase=True):
    # butterworth based on IIR filter is used
    [b, a] = get_filter_coefficients(TIME_PERIOD)
    # zero phase filter is used to generate the labels (slower but no distortion)
    if zero_phase:
        filtered_data = signal.filtfilt(b, a, data)
    # has distortion in the beginning and end of the data but is usable in real-time
    elif not zero_phase:
        filtered_data = signal.lfilter(b, a, data)
    else:
        print("Error: Invalid boolean input!")
        exit()
    return np.ndarray.tolist(filtered_data)  # converts np array to list


def get_filter_coefficients(TIME_PERIOD):
    nyquist = 1 / (2 * TIME_PERIOD)
    cut_off = 5 / nyquist

    # 2nd order IIR butterworth filter is used
    return signal.butter(2, cut_off, btype='lowpass')
