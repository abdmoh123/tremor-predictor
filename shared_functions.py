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
        for j in range(horizon):
            # ensures that index does not go out of bounds
            if i < j:
                break
            else:
                temp_array.append(feature_list[i - j])
        avg_array.append(sum(temp_array) / horizon)  # saves average of the [horizon] values to the array
    return avg_array


# normalises a list to be between -1 and 1
def normalise(feature):
    max_value = find_highest(feature)  # finds the largest value in array
    return [value / max_value for value in feature]  # normalises the values to be between -1 and 1


# iterates through list to find the highest value
def find_highest(feature_list):
    max_value = 0
    for value in feature_list:
        if abs(value) > max_value:
            max_value = abs(value)
    return max_value


# calculates the accuracy of the model using root mean squared error
def calc_accuracy(predicted, actual_output):
    rms_error = mean_squared_error(actual_output, predicted, squared=False)
    rms_output = np.sqrt(sum(np.square(actual_output)))  # calculates the rms of the voluntary motion
    return 100 * (1 - (rms_error / rms_output))


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
