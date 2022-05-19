import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# plots the regression model (works with tremor component)
def plot_model(time, data, data_labels, title):
    fig, axes = plt.subplots(len(data), figsize=(10, 10))

    # plots data
    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            axes[i].plot(time, data[i][j], label=data_labels[j], linewidth=0.5)
        axes[i].set(ylabel=data[i][len(data[0]) - 1])
    axes[len(data) - 1].set(xlabel="Time (s)")

    # for figure legend
    font_prop = FontProperties()
    font_prop.set_size("small")  # font size of the legend content
    lines, labels = axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, prop=font_prop)
    fig.suptitle(title, fontweight="bold")

    # displays graphs
    plt.show()


def plot_accuracies(time, data, data_labels, x_label="", y_label=""):
    fig, axes = plt.subplots(len(data))

    # plots data
    for i in range(len(data)):
        for j in range(len(data[i])):
            axes[i].plot(time, data[i][j], label=data_labels[i][j], linewidth=0.5)
        axes[i].set(ylabel=y_label)
        axes[i].legend()
    axes[len(data) - 1].set(xlabel=x_label)  # sets x label at the bottom subplot

    # displays graphs
    plt.show()


# plots the data (including features) in vertical subplots
def plot_data(time, data, x_label="", y_label=""):
    # plots 1 piece of data
    if len(data) == 1:
        plt.plot(time, data[0][0], label=data[0][1], linewidth=0.5)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend()
    # plots multiple pieces of data
    else:
        fig, axes = plt.subplots(len(data))
        # plots data
        for i in range(len(data)):
            axes[i].plot(time, data[i][0], label=data[i][1], linewidth=0.5)
            axes[i].set(ylabel=y_label)
            axes[i].legend()
        # axes labels and legend
        axes[len(data) - 1].set(xlabel=x_label)

    # displays graphs
    plt.show()
