import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error


def check_bounds(l_bound, u_bound, rows):
    # ensures that the lower bound is smaller than the upper bound
    if l_bound >= u_bound:
        l_bound = u_bound - 1
    # checks if bounds are valid (prevents index out of bounds error)
    if (l_bound < 0) | (l_bound >= len(rows)):
        l_bound = 0
    if (u_bound > len(rows)) | (u_bound <= 0):
        u_bound = len(rows)
    return [l_bound, u_bound]


# calculates the average of every [horizon] values in an array
def calc_average(feature_list, horizon):
    temp_array = []
    avg_array = []
    for i in range(len(feature_list)):
        # for loop below puts the [horizon] previous values in an array (including current value)
        for j in range(i - horizon, i + 1):
            # ensures that index does not go out of bounds
            if j >= 0:
                temp_array.append(feature_list[j])
        avg_array.append(sum(temp_array) / len(temp_array))  # saves average to the array
    return avg_array


# normalises a list to be between -1 and 1
def normalise(features):
    max_value = find_highest(features, True)  # finds the largest value (magnitude) in array
    mean = sum(features) / len(features)  # finds the mean of the array
    return [(value - mean) / max_value for value in features]  # normalises the values to be between -1 and 1


# iterates through list to find the highest value
def find_highest(features, magnitude):
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


def find_lowest(features, magnitude):
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
    # finds the maximum and minimum values in the actual output (labels)
    actual_max = find_highest(actual_output, False)
    actual_min = find_lowest(actual_output, False)
    actual_range = (actual_max - actual_min) / 2  # calculates the range (sigma)

    rms_error = mean_squared_error(actual_output, predicted, squared=False)  # calculates the RMS error
    return 100 * (1 - (rms_error / actual_range))  # uses the range to convert the RMS error into a percentage


# finds the change in tremor output
def calc_delta(time, feature):
    delta_x = []
    t = time[1] - time[0]  # gets the time increment (delta t)
    for i in range(len(feature)):
        # if statement prevents index out of bounds exception
        if i > 0:
            delta_x.append((feature[i] - feature[i - 1]) / t)
        else:
            delta_x.append(feature[i] / t)
    return delta_x


# finds the optimum regularisation parameter value for an SVM regression model
def optimise_reg(features, labels):
    C = 0  # regularisation parameter
    choices = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]  # regularisation parameter values to be tested
    accuracy = 0  # initialised as 0 because it will only go up

    # loops through all the C choices to find the most accurate result
    for temp_C in choices:
        # SVM with rbf kernel and a chosen regularisation parameter
        regression = svm.SVR(kernel="rbf", C=temp_C)
        regression.fit(features, labels)
        predictions = regression.predict(features)

        # calculates the percentage accuracy of the model
        temp_accuracy = calc_accuracy(predictions, labels)

        # C values are only updated if the new value gives a more accurate model
        if temp_accuracy > accuracy:
            accuracy = temp_accuracy
            C = temp_C
    return C
