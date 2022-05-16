# libraries imported
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from predict_folder import predict_dir


def predict_dirs(model_type):
    directory_name = str(pathlib.Path(__file__).parent.resolve()) + "/data/"
    # list holding all folders to use
    folder_names = [
        "Novice Pointing",
        "Novice Tracing",
        "Surgeon Pointing",
        "Surgeon Tracing"
    ]

    # finds the directories
    directories = []
    for folder_name in folder_names:
        os.fsencode(directory_name + folder_name + "/")

    # puts all txt files' names in a list
    file_names = []
    for directory in directories:
        for file in os.listdir(directory):
            file_names.append(os.fsdecode(file))

    all_r2 = []
    all_tremor_r2 = []
    all_nrmse = []
    all_tremor_nrmse = []
    all_training_times = []
    all_prediction_times = []
    # runs the prediction code for each folder
    for folder_name in folder_names:
        [r2_scores, tremor_r2_scores, nrmses, tremor_nrmses, training_times, prediction_times] \
            = predict_dir(directory_name + folder_name, model_type)
        all_r2.append(r2_scores)
        all_tremor_r2.append(tremor_r2_scores)
        all_nrmse.append(nrmses)
        all_tremor_nrmse.append(tremor_nrmses)
        all_training_times.append(training_times)
        all_prediction_times.append(prediction_times)

    # prints the average metrics for all datasets
    print(
        "\nAverage R2 score of the model:", str(np.mean(all_r2)) + "%",
        "\nAverage R2 score of the tremor component:", str(np.mean(all_tremor_r2)) + "%",
        "\nAverage Normalised RMS error of the model:", np.mean(all_nrmse),
        "\nAverage Normalised RMS error of the tremor component:", np.mean(all_tremor_nrmse),
        "\nAverage time taken to train:", str(np.mean(all_training_times)) + "s",
        "\nAverage time taken to make a prediction:", str(np.mean(all_prediction_times)) + "s"
    )
    # data for plotting bar chart
    labels = ["Overall R2", "Tremor R2"]

    if len(folder_names) <= 2:
        fig, axes = plt.subplots(1, len(folder_names))
    else:
        fig, axes = plt.subplots(round(len(folder_names) / 2), round(len(folder_names) / 2))
    # bar chart properties
    bar_width = 0.1
    x_axis = np.arange(len(labels))

    r = c = 0  # row and column indices of subplot
    # run for every folder
    for i in range(len(folder_names)):
        x_axis_r2 = [all_r2[i][0], all_tremor_r2[i][0]]
        y_axis_r2 = [all_r2[i][1], all_tremor_r2[i][1]]
        z_axis_r2 = [all_r2[i][2], all_tremor_r2[i][2]]
        average_r2 = [np.mean(all_r2[i]), np.mean(all_tremor_r2[i])]

        if len(folder_names) <= 2:
            # bars for each result
            axes[i].bar(x_axis - (3 * bar_width / 2), x_axis_r2, width=bar_width, label="X")
            axes[i].bar(x_axis - (bar_width / 2), y_axis_r2, width=bar_width, label="Y")
            axes[i].bar(x_axis + (bar_width / 2), z_axis_r2, width=bar_width, label="Z")
            axes[i].bar(x_axis + (3 * bar_width / 2), average_r2, width=bar_width, label="Average")

            # axis labels + title
            axes[i].set_xlabel("R2 score metrics")
            axes[i].set_ylabel("Accuracy (%)")
            # setting ticks
            axes[i].set_xticks(x_axis)
            axes[i].set_xticklabels(labels)
            # legend
            axes[i].legend(title="3D axis")
            # tick parameters
            axes[i].tick_params(axis="x", which="both")
            axes[i].tick_params(axis="y", which="both")
        else:
            # allows subplots to be displayed in a grid-like shape
            if r >= len(folder_names) / 2:
                r = 0
                c += 1

            # bars for each result
            axes[r, c].bar(x_axis - (3 * bar_width / 2), x_axis_r2, width=bar_width, label="X")
            axes[r, c].bar(x_axis - (bar_width / 2), y_axis_r2, width=bar_width, label="Y")
            axes[r, c].bar(x_axis + (bar_width / 2), z_axis_r2, width=bar_width, label="Z")
            axes[r, c].bar(x_axis + (3 * bar_width / 2), average_r2, width=bar_width, label="Average")

            # axis labels + title
            axes[r, c].set_title(folder_names[i])
            axes[r, c].set_xlabel("R2 score metrics")
            axes[r, c].set_ylabel("Accuracy (%)")
            # setting ticks
            axes[r, c].set_xticks(x_axis)
            axes[r, c].set_xticklabels(labels)
            # legend
            axes[r, c].legend(title="3D axis")
            # tick parameters
            axes[r, c].tick_params(axis="x", which="both")
            axes[r, c].tick_params(axis="y", which="both")

            r += 1  # moves on to the next row in the subplot
    fig.suptitle(model + " results based on multiple datasets")
    plt.show()


if __name__ == '__main__':
    model = "SVM"
    # model = "Random Forest"

    predict_dirs(model)
