# libraries imported
import csv
import matplotlib.pyplot as plt

from shared_functions import *


def main():
    file_name = "./simulated_tremor_data.csv"
    horizon = 5  # amount of data to be temporarily stored for feature creation

    # training data
    data = read_data(file_name, 0, 152, 1)  # simulated tremor data
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
    C = optimise_reg(X, y)
    C = 3  # optimum C value = 3
    print("Regularisation parameter C:", C)

    # SVM with rbf kernel
    regression = svm.SVR(kernel="rbf", C=C)
    regression.fit(X, y)
    predictions = regression.predict(X)

    # calculates and prints the RMSE of the model
    accuracy = calc_accuracy(predictions, y)
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
        [l_bound, u_bound] = check_bounds(l_bound, u_bound, rows)

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
