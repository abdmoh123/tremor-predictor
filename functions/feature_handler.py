import math

import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error

import functions.miscellaneous as mf


# finds the change in tremor output
def calc_delta(time, feature):
    delta_x = []
    t = time[1] - time[0]  # gets the time increment (delta t)
    for i in range(len(feature)):
        # if statement prevents index out of bounds exception
        if i > 0:
            delta_x.append((feature[i] - feature[i - 1]) / t)
        else:
            delta_x.append(0)
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


# finds the best value for how far back the past motion feature should be
def optimise_past_motion(features, labels, max_shift_value=10, C=1):
    input_motion = features[0]
    rms_error = 0
    shift_value = 1

    temp_shift_value = shift_value
    while temp_shift_value <= max_shift_value:
        # shifts the input motion by a set amount to get the past motion as a feature
        shifted_input = shift(input_motion, temp_shift_value)

        # puts the features (including shifted_input) in an array ready for SVR fitting
        temp_features = list(features)
        temp_features.append(shifted_input)
        temp_features = np.vstack(temp_features).T

        # SVM with rbf kernel
        regression = svm.SVR(kernel="rbf", C=C)
        regression.fit(temp_features, labels)
        predictions = regression.predict(temp_features)

        # calculates the rms error of the model
        temp_rmse = mean_squared_error(input_motion, predictions, squared=False)  # calculates the RMS error
        # shift value is only updated if the new value gives a more accurate model
        if temp_rmse < rms_error:
            rms_error = temp_rmse
            shift_value = temp_shift_value
        temp_shift_value += 1  # increments the shift value
    return shift_value  # returns the optimal shift value


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


# shifts values in an array using np.roll
def shift(data, shift_value):
    new_data = np.roll(data, shift_value)
    # fills up new shifted slots with 0 (beginning or end of array)
    if shift_value > 0:
        np.put(new_data, range(shift_value), 0)  # fills the beginning
    elif shift_value < 0:
        np.put(new_data, range(len(new_data) - shift_value, len(new_data)), 0)  # fills the end
    return new_data
