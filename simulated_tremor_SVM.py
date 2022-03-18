# libraries imported
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# functions that apply to both simulated and real tremor
import functions.feature_handler as fh
import functions.miscellaneous as mf


def main():
    file_name = "./data/simulated_tremor_data.csv"

    # training data
    data = read_data(file_name, 0, 152, 1)  # simulated tremor data
    x1 = data[0]  # horizontal component (time)
    x2 = fh.normalise(data[1])  # noisy tremor output + normalises it
    [y, y_mean, y_sigma] = fh.normalise(data[2], True)  # intended movement (for training) + normalises it
    # prints data in the console
    print("X1:\n", x1)
    print("X2:\n", x2)
    print("Y:\n", y)

    # calculates the change in tremor output + normalises it
    delta_x = fh.normalise(fh.calc_delta(x1, x2))
    print("Change in X:\n", delta_x)

    # finds the optimum value for C (regularisation parameter)
    [horizon, C] = mf.optimise([x2, delta_x], y)
    # C = 2.43  # optimal C value = 2.43
    # horizon = x  # optimal horizon value = 5
    print("Regularisation parameter C:", C, "\nHorizon value:", horizon)

    # calculates average of every [horizon] tremor outputs + normalises it
    avg_x = fh.normalise(fh.calc_average(x2, horizon))
    print("Average X values:\n", avg_x)

    # puts the features in 1 array
    X = np.vstack((x2, delta_x, avg_x)).T

    # SVM with rbf kernel
    regression = svm.SVR(kernel="rbf", C=C)
    regression.fit(X, y)
    predictions = fh.denormalise(regression.predict(X), y_mean, y_sigma)

    # calculates and prints the RMSE of the model
    y = fh.denormalise(y, y_mean, y_sigma)
    accuracy = mf.calc_accuracy(predictions, y)
    print("\nAccuracy: " + str(accuracy) + "%")

    # plots the data and model on a graph
    plot_model(x1, data, predictions)


# plots the regression model and inputted data on graphs
def plot_model(time, data, predictions):
    # splits plot window into 2 graphs
    fig, axes = plt.subplots(2)

    # plots data
    axes[0].plot(time, data[1], label="Noisy data with tremor")
    axes[0].plot(time, data[2], label="Intended movement without tremor")
    axes[0].set(ylabel="X motion voltage (V)")
    axes[0].legend()

    # plots SVM regression model
    axes[1].plot(time, predictions, label="SVM regression model")
    axes[1].plot(time, data[2], label="Intended movement without tremor")
    axes[1].set(ylabel="X motion voltage (V)")
    axes[1].set(xlabel="time/index")
    axes[1].legend()

    # displays the plots
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

        # ensures bounds are valid
        [l_bound, u_bound] = mf.check_bounds(l_bound, u_bound, rows)

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


if __name__ == '__main__':
    main()
