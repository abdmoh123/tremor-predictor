import math

import numpy as np

import functions.miscellaneous as mf


# finds the change in tremor output
def calc_delta(time, feature, index_difference=1):
    delta_x = []
    t = (time[1] - time[0]) * index_difference  # gets the time increment (delta t)
    for i in range(len(feature)):
        # if statement prevents index out of bounds exception
        if i > (index_difference - 1):
            delta_x.append((feature[i] - feature[i - index_difference]) / t)
        else:
            delta_x.append(feature[i] - feature[0])
    return delta_x


# calculates the average of every [horizon] values in an array
def calc_average(features, horizon):
    avg_array = []
    for i in range(len(features)):
        # ensures the average is still calculated correctly at the beginning of the feature list
        if (2 * i) < (horizon - 1):
            temp_array = features[0:(2 * i + 1)]
        else:
            # the correct values are selected (i in the middle) even if the horizon is even
            if horizon % 2 == 0:
                horizon_delta = int(math.floor(horizon / 2))
                temp_array = features[(i - horizon_delta):(i + horizon_delta)]
            else:
                horizon_delta = int(math.floor(horizon / 2))
                temp_array = features[(i - horizon_delta):(i + horizon_delta + 1)]
        avg_array.append(sum(temp_array) / len(temp_array))  # saves average to the array
    return avg_array


# shifts values in an array using np.roll
def shift(data, shift_value=1):
    # prevents index out of bounds error while performing the same function
    if shift_value > len(data):
        shift_value -= len(data)

    new_data = np.roll(data, shift_value)
    # fills up new shifted slots with the first or last element value (beginning or end of array)
    if shift_value > 0:
        first_element = new_data[shift_value]
        np.put(new_data, range(shift_value), first_element)  # fills the beginning
    elif shift_value < 0:
        last_element = new_data[len(new_data) + shift_value]
        np.put(new_data, range(len(new_data) - shift_value, len(new_data)), last_element)  # fills the end
    return new_data


def divide_data(data, index_difference=1):
    divided_data = []
    for i in range(len(data)):
        # if statement prevents index out of bounds exception
        if i > (index_difference - 1):
            divided_data.append((data[i] / (data[i - index_difference] + 0.01)))  # 0.01 to prevent division by 0
        else:
            divided_data.append(data[i] / (data[0] + 0.01))  # 0.01 to prevent division by 0
    return divided_data


# normalises a list to be between -1 and 1
def normalise(data, return_averages=False):
    sigma = (mf.find_highest(data) - mf.find_lowest(data)) / 2  # calculates the standard deviation (range / 2)
    mean = sum(data) / len(data)  # finds the mean of the array
    norm_data = [(value - mean) / sigma for value in data]  # normalises the values to be between -1 and 1

    # returns the mean and spread if the function call specified
    if return_averages:
        return norm_data, mean, sigma
    return norm_data


# reverses the normalisation
def denormalise(data, mean, sigma):
    return [(value * sigma) + mean for value in data]