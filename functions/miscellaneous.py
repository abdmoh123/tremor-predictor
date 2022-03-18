import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error

import functions.feature_handler as fh


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


# iterates through list to find the highest value
def find_highest(features, magnitude=False):
    max_value = 0
    # can find the largest magnitude or the most positive value
    if magnitude:
        for value in features:
            if abs(value) > max_value:
                max_value = abs(value)
    else:
        for value in features:
            if value > max_value:
                max_value = value
    return max_value


def find_lowest(features, magnitude=False):
    min_value = features[0]
    # can find the smallest magnitude or the most negative value
    if magnitude:
        for value in features:
            if abs(value) < abs(min_value):
                min_value = abs(value)
    else:
        for value in features:
            if value < min_value:
                min_value = value
    return min_value


# calculates the accuracy of the model using root mean squared error
def calc_accuracy(predicted, actual_output):
    rms_error = mean_squared_error(actual_output, predicted, squared=False)  # calculates the RMS error
    sigma = np.std(actual_output)  # calculates the standard deviation of voluntary motion
    return 100 * (1 - (rms_error / sigma))  # uses the standard deviation to convert the RMS error into a percentage


# finds the optimal regularisation parameter and average horizon for an SVM regression model
def optimise(features, labels):
    horizon = 1  # regularisation parameter (starts at 1 to prevent division by zero)
    max_horizon = 50  # limit for the horizon loop
    horizon_increment = 2

    C = 0.01  # regularisation parameter
    max_C = 30  # limit for the C loop
    C_increment = 3

    accuracy = 0  # initialised as 0 because it will only go up

    temp_horizon = horizon  # temp value for iteration
    # loop evaluates models repeatedly to find optimal horizon value
    while temp_horizon <= max_horizon:
        temp_C = 0.01  # temp value for iteration
        # loops to find optimal regularisation parameter value (C)
        while temp_C <= max_C:
            # calculates the average motion
            average = fh.normalise(fh.calc_average(features[0], temp_horizon))

            # puts the features (including average) in an array ready for SVR fitting
            temp_features = list(features)
            temp_features.append(average)
            temp_features = np.vstack(temp_features).T

            # SVM with rbf kernel
            regression = svm.SVR(kernel="rbf", C=temp_C)
            regression.fit(temp_features, labels)
            predictions = regression.predict(temp_features)

            # calculates the percentage accuracy of the model
            temp_accuracy = calc_accuracy(predictions, labels)

            # horizon is only updated if the new value gives a more accurate model
            if temp_accuracy >= accuracy:
                accuracy = temp_accuracy
                horizon = temp_horizon
                C = temp_C
            temp_C *= C_increment  # C is multiplied with every loop (to give estimate optimum)
        temp_horizon += horizon_increment  # horizon values are incremented in values of 2
    return horizon, C
