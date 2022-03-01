# libraries imported
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error
from scipy import signal


def main():
    file_name = "real_tremor_data.csv"
    horizon = 5  # amount of data to be temporarily stored for feature creation

    # reads data into memory
    data = read_data(file_name, 200, 4000, 4)  # real tremor data (t, x, y, z, grip force)
    filtered_data = filter_data(data)  # filters data to get an estimate of intended movement
    t = data[0]  # horizontal component (time)
    x_motion = data[1]  # tremor in x axis
    x_label = filtered_data[1]  # intended movement in x axis
    y_motion = data[2]  # tremor in y axis
    y_label = filtered_data[2]  # intended movement in y axis
    z_motion = data[3]  # tremor in z axis
    z_label = filtered_data[3]  # intended movement in z axis

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


# reads data in a csv file and puts them in a 2D list
def read_data(file_name, l_bound, u_bound, label_col):
    # data will be separated in these lists
    features = []
    labels = []

    with open(file_name, "r") as file:
        reader = csv.reader(file)
        rows = list(reader)

        # ensures that the lower bound is smaller than the upper bound
        if l_bound >= u_bound:
            l_bound = u_bound - 1
        # checks if bounds are valid (prevents index out of bounds error)
        if (l_bound < 0) | (l_bound >= len(rows)):
            l_bound = 0
        if (u_bound > len(rows)) | (u_bound <= 0):
            u_bound = len(rows)

        # reads through the file and puts the data in the respective lists above
        for i in range(l_bound, u_bound):
            row = rows[i]
            temp_features = []
            # separates the labels from the features
            for j in range(len(row)):
                column = float(row[j])
                if j == label_col:
                    labels.append(column)
                else:
                    temp_features.append(column)
            # lists all the features in a 2D list
            features.append(temp_features)
    # combines the 2 lists into a 2D numpy array with each feature/label being its own sub-array
    features = np.vstack(features).T
    return np.append(features, [labels], axis=0)


def filter_data(data):
    time_period = 1 / 250
    nyquist = 1 / (2 * time_period)
    cut_off = 5 / nyquist

    [b, a] = signal.butter(2, cut_off, btype='low')
    return signal.filtfilt(b, a, data)


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


# finds the change in tremor output
def calc_delta(x1, x2):
    delta_x = []
    t = x1[1]  # gets the time increment (delta t)
    for i in range(len(x2)):
        # if statement prevents index out of bounds exception
        if i > 0:
            delta_x.append((x2[i] - x2[i - 1]) / t)
        else:
            delta_x.append(x2[i] / t)
    return delta_x


# calculates the accuracy of the model using root mean squared error
def calc_accuracy(predicted, actual_output):
    rms_error = mean_squared_error(actual_output, predicted, squared=False)
    rms_output = np.sqrt(sum(np.square(actual_output)))  # calculates the rms of the voluntary motion
    return 100 * (1 - (rms_error / rms_output))


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


if __name__ == '__main__':
    main()
