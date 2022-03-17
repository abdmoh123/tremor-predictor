import math
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
def calc_average(feature_list, horizon):
    avg_array = []
    for i in range(len(feature_list)):
        # ensures the average is still calculated correctly at the beginning of the feature list
        if (2 * i) < (horizon - 1):
            temp_array = feature_list[0:(2 * i + 1)]
        else:
            # the correct values are selected (i in the middle) even if the horizon is even
            if horizon % 2 == 0:
                horizon_delta = int(math.floor(horizon / 2))
                temp_array = feature_list[(i - horizon_delta):(i + horizon_delta)]
            else:
                horizon_delta = int(math.floor(horizon / 2))
                temp_array = feature_list[(i - horizon_delta):(i + horizon_delta + 1)]
        avg_array.append(sum(temp_array) / len(temp_array))  # saves average to the array
    return avg_array


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
