# libraries imported
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error


def main():
    file_name = "tremor_training_data.csv"
    # file_name = "real_tremor_data.csv"
    horizon = 5  # amount of data to be temporarily stored for feature creation

    # training data
    data = read_data(file_name, 0, 152, 1)  # simulated tremor data
    # data = read_data(file_name, 200, 4000, 1)  # real tremor data
    x1 = data[0]  # horizontal component (time)
    x2 = data[1]  # noisy tremor output
    y = data[2]  # intended movement (for training) - labels will always be the last column in array
    # prints training data in the console
    print("X1:\n", x1)
    print("X2:\n", x2)
    print("Y:\n", y)

    # calculates the change in tremor output
    delta_x = calc_delta(x1, x2)
    print("Change in X:\n", delta_x)

    # calculates average of every [horizon] tremor outputs
    avg_x = calc_average(x2, horizon)
    print("Average X values:\n", avg_x)

    # applies feature scaling to training data
    x2 = normalise(x2)
    delta_x = normalise(delta_x)
    avg_x = normalise(avg_x)

    # puts the features in 1 array
    X = np.vstack((x2, delta_x, avg_x)).T

    # finds the optimum value for C (regularisation parameter)
    # C = optimise_reg(X, y)
    # print("Regularisation parameter C:", C)

    # SVM with rbf kernel
    regression = svm.SVR(kernel="rbf", C=3)  # optimum C value = 3
    regression.fit(X, y)
    predictions = regression.predict(X)

    # calculates and prints the percentage accuracy of the model
    accuracy = calc_accuracy(predictions, y)
    print("\nAccuracy: " + str(accuracy) + "%")

    plot_model(data, predictions)


# plots the regression model and inputted data on graphs
def plot_model(data, predictions):
    # splits the graph into 2 subplots
    fig, axes = plt.subplots(2)

    # plots training data in graph
    axes[0].plot(data[0], data[1], label="Training: Noisy tremor")
    axes[0].plot(data[0], data[2], label="Training: Intended movement")
    axes[0].legend()
    # plots SVM regression model
    axes[1].scatter(data[0], data[1], s=5, label="Noisy data with tremor")
    axes[1].scatter(data[0], data[2], s=5, label="Intended movement without tremor")
    axes[1].plot(data[0], predictions, label="SVM regression model")
    axes[1].legend()

    # displays the plots
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
