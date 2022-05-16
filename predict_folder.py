# libraries imported
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from real_time_predictor import start_predictor


def predict_dir(path, model_type):
    directory = os.fsencode(path)

    # puts all txt files' names in a list
    file_names = []
    for file in os.listdir(directory):
        file_names.append(os.fsdecode(file))

    r2 = []
    tremor_r2 = []
    nrmse = []
    tremor_nrmse = []
    max_training_times = []
    avg_prediction_times = []
    # runs predictor algorithm for each dataset
    for file_name in file_names:
        [accuracy, tremor_accuracy, max_training_time, avg_prediction_time] \
            = start_predictor(directory_name + "/" + file_name, model_type)
        r2.append(accuracy[0])
        tremor_r2.append(tremor_accuracy[0])
        nrmse.append(accuracy[1])
        tremor_nrmse.append(tremor_accuracy[1])
        max_training_times.append(max_training_time)
        avg_prediction_times.append(avg_prediction_time)
    return r2, tremor_r2, nrmse, tremor_nrmse, max_training_times, avg_prediction_times


if __name__ == '__main__':
    # model = "SVM"
    model = "Random Forest"
    folder_name = "Surgeon Pointing"

    # finds the directory
    directory_name = str(pathlib.Path(__file__).parent.resolve()) + "/data/" + folder_name

    [r2_scores, tremor_r2_scores, nrmses, tremor_nrmses, training_times, prediction_times] \
        = predict_dir(directory_name, model)

    # finds and prints the average metrics for all datasets
    overall_r2_score_3D = np.mean(r2_scores, axis=0)
    overall_nrmse_3D = np.mean(nrmses, axis=0)
    overall_tremor_r2_score_3D = np.mean(tremor_r2_scores, axis=0)
    overall_tremor_nrmse_3D = np.mean(tremor_nrmses, axis=0)
    overall_training_time = np.mean(training_times, axis=0)
    overall_avg_prediction_time_3D = np.mean(prediction_times, axis=0)
    print(
        "\nAverage R2 score of the model (%):", overall_r2_score_3D,
        "\nAverage Normalised RMS error of the model:", overall_nrmse_3D,
        "\nAverage R2 score of the tremor component (%):", overall_tremor_r2_score_3D,
        "\nAverage Normalised RMS error of the tremor component:", overall_tremor_nrmse_3D,
        "\nAverage time taken to train (s):", overall_training_time,
        "\nAverage time taken to make a prediction (s)", overall_avg_prediction_time_3D
    )

    # data for plotting bar chart
    labels = ["Overall R2 score", "Tremor component R2 score"]
    x_axis_r2 = [overall_r2_score_3D[0], overall_tremor_r2_score_3D[0]]
    y_axis_r2 = [overall_r2_score_3D[1], overall_tremor_r2_score_3D[1]]
    z_axis_r2 = [overall_r2_score_3D[2], overall_tremor_r2_score_3D[2]]
    average_r2 = [np.mean(overall_r2_score_3D), np.mean(overall_tremor_r2_score_3D)]

    fig, axis = plt.subplots()
    # bar chart properties
    bar_width = 0.1
    x_axis = np.arange(len(labels))

    # bars for each result
    bar1 = axis.bar(x_axis - (3 * bar_width / 2), x_axis_r2, width=bar_width, label="X")
    bar2 = axis.bar(x_axis - (bar_width / 2), y_axis_r2, width=bar_width, label="Y")
    bar3 = axis.bar(x_axis + (bar_width / 2), z_axis_r2, width=bar_width, label="Z")
    bar4 = axis.bar(x_axis + (3 * bar_width / 2), average_r2, width=bar_width, label="Average")

    # axis labels + title
    axis.set_xlabel("R2 score metrics")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title(model + " results based on the " + folder_name + " datasets")
    # setting ticks
    axis.set_xticks(x_axis)
    axis.set_xticklabels(labels)
    # legend
    axis.legend(title="3D axis")
    # tick parameters
    axis.tick_params(axis="x", which="both")
    axis.tick_params(axis="y", which="both")

    plt.show()
