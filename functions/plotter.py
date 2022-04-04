import matplotlib.pyplot as plt


# plots the regression model (works with tremor component)
def plot_model(time, data, axes_labels):
    fig, axes = plt.subplots(len(data))

    # plots data
    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            axes[i].plot(time, data[i][j], label=axes_labels[j], linewidth=0.5)
        axes[i].set(ylabel=data[i][len(data[0]) - 1])
        axes[i].legend()
    axes[len(data) - 1].set(xlabel="Time (s)")  # sets x label at the bottom subplot

    # displays graphs
    plt.show()


# plots the features in vertical subplots
def plot_features(time, data, y_axis_label):
    fig, axes = plt.subplots(len(data))

    # plots data
    for i in range(len(data)):
        axes[i].plot(time, data[i][0], label=data[i][1], linewidth=0.5)
        axes[i].set(ylabel=y_axis_label)
        axes[i].legend()

    # axes labels and legend
    axes[len(data) - 1].set(xlabel="Time (s)")

    # displays graphs
    plt.show()
