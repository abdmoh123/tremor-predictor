# libraries imported
import os
import numpy as np
import matplotlib.pyplot as plt
from real_time_predictor import start_predictor


def predict_dir(path, model_type):
    directory = os.fsencode(path)

    # puts all txt files' names in a list
    file_names = []
    for file in os.listdir(directory):
        file_names.append(os.fsdecode(file))

    r2_scores = []
    nrmses = []
    tremor_r2_scores = []
    tremor_nrmses = []
    training_times = []
    all_prediction_times = []
    # runs predictor algorithm for each dataset
    for file_name in file_names:
        [accuracy, tremor_accuracy, max_training_time, avg_prediction_times] \
            = start_predictor(directory_name + "/" + file_name, model_type)
        r2_scores.append(accuracy[0])
        nrmses.append(accuracy[1])
        tremor_r2_scores.append(tremor_accuracy[0])
        tremor_nrmses.append(tremor_accuracy[1])
        training_times.append(max_training_time)
        all_prediction_times.append(avg_prediction_times)

    # finds and outputs the average metrics for all datasets
    overall_r2_score_3D = np.mean(r2_scores, axis=0)
    overall_nrmse_3D = np.mean(nrmses, axis=0)
    overall_tremor_r2_score_3D = np.mean(tremor_r2_scores, axis=0)
    overall_tremor_nrmse_3D = np.mean(tremor_nrmses, axis=0)
    overall_training_time = np.mean(training_times, axis=0)
    overall_avg_prediction_time_3D = np.mean(all_prediction_times, axis=0)
    print(
        "\nAverage R2 score of the model (%):", overall_r2_score_3D,
        "\nAverage Normalised RMS error of the model:", overall_nrmse_3D,
        "\nAverage R2 score of the tremor component (%):", overall_tremor_r2_score_3D,
        "\nAverage Normalised RMS error of the tremor component:", overall_tremor_nrmse_3D,
        "\nAverage time taken to train (s):", overall_training_time,
        "\nAverage time taken to make a prediction (s)", overall_avg_prediction_time_3D
    )
    bar_prediction_times = [[], [], []]
    bar_r2 = [[], [], []]
    bar_tremor_r2 = [[], [], []]
    for i in range(len(all_prediction_times)):
        for j in range(len(all_prediction_times[i])):
            bar_prediction_times[j] = all_prediction_times[i][j]
            bar_r2[j] = r2_scores[i][j]
            bar_tremor_r2[j] = tremor_r2_scores[i][j]

    index = np.arange(len(r2_scores))
    fig, axes = plt.subplots(4)
    dimension_label = ["X", "Y", "Z"]

    axes[0].bar(index, training_times, label="Training time")
    for i in range(len(bar_prediction_times)):
        axes[1].bar(index, bar_prediction_times[i], label="Prediction time " + dimension_label[i])
        axes[2].bar(index, bar_r2[i], label="Overall accuracy " + dimension_label[i])
        axes[3].bar(index, bar_tremor_r2[i], label="Tremor accuracy " + dimension_label[i])
    axes[0].set_ylabel("Time (s)")
    axes[1].set_ylabel("Time (s)")
    axes[2].set_ylabel("Score (%)")
    axes[3].set_ylabel("Score (%)")
    for i in range(4):
        axes[i].legend()

    plt.show()


if __name__ == '__main__':
    # model = "SVM"
    model = "Random Forest"

    # finds the directory
    folder_name = "Surgeon Pointing"
    directory_name = "C:/Users/Abdul/OneDrive - Newcastle University/Stage 3/Obsidian Vault/" \
                     "EEE3095-7 Individual Project and Dissertation/Tremor ML/data/" + folder_name  # desktop
    # directory_name = "C:/Users/abdha/OneDrive - Newcastle University/Stage 3/Obsidian Vault/" \
    #                  "EEE3095-7 Individual Project and Dissertation/Tremor ML/data/" + folder_name  # laptop

    predict_dir(directory_name, model)
