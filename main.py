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
    X = []

    # testing data
    x1_test = []  # horizontal component (time)
    x2_test = []  # noisy tremor output
    y_test = []  # intended movement to predict

    # reads training csv file
    with open("tremor_training_data.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            x1.append(float(row[0]))
            y.append(float(row[1]))
            x2.append(float(row[2]))
            X.append([float(row[0]), float(row[2])])

    # prints data in the console
    print("X1:\n", x1)
    print("X2:\n", x2)
    print("Y:\n", y)

    # reads testing csv file
    with open("tremor_test_data.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            x1_test.append(float(row[0]))
            y_test.append(float(row[1]))
            x2_test.append(float(row[2]))

    # prints data in the console
    print("X1 test:\n", x1_test)
    print("X2 test:\n", x2_test)
    print("Y test:\n", y_test)

    # calculates the change in tremor output
    delta_x = find_delta(x1, x2)
    print("Change in X:\n", delta_x)

    # applies feature scaling
    x2 = normalise(x2)
    delta_x = normalise(delta_x)
    # applies feature scaling to test data
    x2_test = normalise(x2_test)

    X = np.vstack((x2, delta_x)).T
    print("Features:\n", X)

    # reshapes the input to the correct format (1 feature with many samples)
    # x2 = np.array(x2).reshape(-1, 1)
    # x2_test = np.array(x2_test).reshape(-1, 1)

    # SVM with rbf kernel
    regression = svm.SVR(kernel="rbf")
    regression.fit(X, y)
    predictions = regression.predict(X)

    fig, axes = plt.subplots(3)

    # plots training data in graph
    axes[0].plot(x1, x2, label="Training: Noisy tremor")
    axes[0].plot(x1, y, label="Training: Intended movement")
    axes[0].legend()

    # plots testing data in graph
    axes[1].plot(x1_test, x2_test, label="Testing: Noisy tremor")
    axes[1].plot(x1_test, y_test, label="Testing: Intended movement")
    axes[1].legend()

    # plots SVM regression model
    axes[2].scatter(x1, x2, s=5, label="Noisy data with tremor")
    axes[2].scatter(x1, y, s=5, label="Intended movement without tremor")
    axes[2].plot(x1, predictions, label="SVM regression model")
    axes[2].legend()

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


# calculates the change in tremor output
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


if __name__ == '__main__':
    main()
