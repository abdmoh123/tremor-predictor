# libraries imported
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def main():
    # training data
    x1 = []  # horizontal component (time)
    x2 = []  # noisy tremor output
    y = []  # intended movement (for training)

    # reads training csv file
    with open("tremor_training_data.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            x1.append(float(row[0]))
            y.append(float(row[1]))
            x2.append(float(row[2]))

    # prints training data in the console
    print("X1:\n", x1)
    print("X2:\n", x2)
    print("Y:\n", y)

    # calculates the change in tremor output
    delta_x = find_delta(x1, x2)
    print("Change in X:\n", delta_x)

    # applies feature scaling to training data
    x2 = normalise(x2)
    delta_x = normalise(delta_x)

    # puts the features in 1 array
    X = np.vstack((x2, delta_x)).T

    # SVM with rbf kernel
    regression = svm.SVR(kernel="rbf")
    regression.fit(X, y)
    predictions = regression.predict(X)

    # calculates and prints the percentage accuracy of the model
    accuracy = calculate_accuracy(predictions, y)
    print("\nAccuracy: " + str(accuracy) + "%")

    # splits the graph into 2 subplots
    fig, axes = plt.subplots(2)
    # plots training data in graph
    axes[0].plot(x1, x2, label="Training: Noisy tremor")
    axes[0].plot(x1, y, label="Training: Intended movement")
    axes[0].legend()
    # plots SVM regression model
    axes[1].scatter(x1, x2, s=5, label="Noisy data with tremor")
    axes[1].scatter(x1, y, s=5, label="Intended movement without tremor")
    axes[1].plot(x1, predictions, label="SVM regression model")
    axes[1].legend()
    # displays the plots
    plt.show()


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
def find_delta(x1, x2):
    delta_x = []
    t = x1[1]  # gets the time increment (delta t)
    for i in range(len(x2)):
        # if statement prevents index out of bounds exception
        if i > 0:
            delta_x.append((x2[i] - x2[i - 1]) / t)
        else:
            delta_x.append(x2[i] / t)
    return delta_x


# calculates how accurate the model is
def calculate_accuracy(predicted, actual_output):
    diff = np.absolute(np.subtract(predicted, actual_output))
    average_diff = np.sum(diff) / len(actual_output)
    return 100 - average_diff  # returns percentage accuracy


if __name__ == '__main__':
    main()
