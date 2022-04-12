import math
import numpy as np
import functions.optimiser as op


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


def gen_features(time, motion, labels=None, horizon=None):
    velocity = []  # feature 2
    acceleration = []  # feature 3
    past_motion = []  # feature 4
    for i in range(len(motion)):
        # calculates the rate of change of 3D motion
        velocity.append(calc_delta(time, motion[i]))
        # calculates the rate of change of rate of change of 3D motion (rate of change of velocity)
        acceleration.append(calc_delta(time, velocity[i]))
        # uses the past data as a feature
        past_motion.append(normalise(shift(motion[i])))  # previous value

        # smoothing the velocity and acceleration
        velocity[i] = normalise(calc_average(velocity[i], 5))
        acceleration[i] = normalise(calc_average(acceleration[i], 5))

    # finds the optimum C and horizon values if no horizon values are inputted
    if (horizon is None) and (labels is not None):
        features = []
        # puts all existing features in a list for model optimisation
        for i in range(len(motion)):
            features.append([
                motion[i],
                velocity[i],
                acceleration[i],
                past_motion[i]
            ])

        # finds the optimum value for horizon
        # print("Optimising horizons...")
        # horizon = []
        # # only required to run once
        # for i in range(len(features)):
        #     horizon.append(op.optimise_parameter(features[i], labels[i], "horizon"))
        # print("Done!")
        # used to save time (optimising is only required once)
        horizon = [30, 30, 30]  # X, Y, Z

        for i in range(len(motion)):
            # calculates the average 3D motion
            average = normalise(calc_average(motion[i], horizon[i]))  # last feature
            # adds the average feature to the features list
            features[i].append(average)
        return features, horizon
    elif horizon is not None:
        features = []
        # puts existing features in a list
        for i in range(len(motion)):
            features.append([
                motion[i],
                velocity[i],
                acceleration[i],
                past_motion[i]
            ])

        for i in range(len(motion)):
            # calculates the average 3D motion
            average = normalise(calc_average(motion[i], horizon[i]))  # last feature
            # adds the average feature to the features list
            features[i].append(average)
        return features
    else:
        # quits the program if an argument is missing
        print("\nMissing argument! (horizon or labels)")
        exit()


# normalises a list to be between -1 and 1
def normalise(data, return_averages=False):
    sigma = (np.max(data) - np.min(data)) / 2  # calculates the spread of the data (range / 2)
    mid = (np.max(data) + np.min(data)) / 2  # finds the midpoint of the data
    norm_data = np.subtract(data, mid) / sigma  # normalises the values to be between -1 and 1

    # returns the mean and spread if the function call specified
    if return_averages:
        return norm_data, mid, sigma
    return norm_data


# reverses the normalisation
def denormalise(data, mid, sigma):
    return np.multiply(data, sigma) + mid
