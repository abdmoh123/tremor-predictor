# libraries imported
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import real_time_voluntary_motion as voluntary_motion_predictor
import real_time_tremor_component as tremor_predictor


def predict_dir(path, model_type, target_prediction):
    directory = os.fsencode(path)

    # puts all txt files' names in a list
    file_names = []
    for file in os.listdir(directory):
        file_names.append(os.fsdecode(file))

    if model_type == "SVM":
        C_parameters = []
        epsilon_parameters = []
    elif model_type == "Random Forest":
        num_trees = []
    else:
        print("Invalid model type!")
        exit()

    r2 = []
    tremor_r2 = []
    nrmse = []
    tremor_nrmse = []
    max_training_times = []
    avg_prediction_times = []
    # runs predictor algorithm for each dataset
    for file_name in file_names:
        if target_prediction == "voluntary motion":
            [model_hyperparameters, accuracy, tremor_accuracy, max_training_time, avg_prediction_time] \
                = voluntary_motion_predictor.start_predictor(path + "/" + file_name, model_type)
        elif target_prediction == "tremor component":
            [model_hyperparameters, accuracy, tremor_accuracy, max_training_time, avg_prediction_time] \
                = tremor_predictor.start_predictor(path + "/" + file_name, model_type)
        else:
            print("Invalid prediction target!")
            exit()

        if model_type == "SVM":
            C_parameters.append(np.array(model_hyperparameters)[:, 0].astype(float))
            epsilon_parameters.append(np.array(model_hyperparameters)[:, 1].astype(float))
        elif model_type == "Random Forest":
            num_trees.append(model_hyperparameters)
        else:
            print("Invalid model type!")
            exit()

        r2.append(accuracy[0])
        tremor_r2.append(tremor_accuracy[0])
        nrmse.append(accuracy[1])
        tremor_nrmse.append(tremor_accuracy[1])
        max_training_times.append(max_training_time)
        avg_prediction_times.append(avg_prediction_time)
    if model_type == "SVM":
        maxmin_hyperparameters = [
            np.max(C_parameters), np.min(C_parameters),
            np.max(epsilon_parameters), np.min(epsilon_parameters)
        ]
    elif model_type == "Random Forest":
        maxmin_hyperparameters = [np.max(num_trees), np.min(num_trees)]
    else:
        print("Invalid model type!")
        exit()

    overall_r2_score = np.mean(r2, axis=0)
    overall_tremor_r2_score = np.mean(tremor_r2, axis=0)
    overall_nrmse = np.mean(nrmse, axis=0)
    overall_tremor_nrmse = np.mean(tremor_nrmse, axis=0)
    overall_training_time = np.mean(max_training_times, axis=0)
    overall_avg_prediction_time = np.mean(avg_prediction_times, axis=0)
    return [
        maxmin_hyperparameters,
        overall_r2_score,
        overall_tremor_r2_score,
        overall_nrmse,
        overall_tremor_nrmse,
        overall_training_time,
        overall_avg_prediction_time
    ]


if __name__ == '__main__':
    # choose what ML regression algorithm to use
    model = "SVM"
    # model = "Random Forest"

    # choose what output to predict
    prediction_target = "voluntary motion"
    # prediction_target = "tremor component"

    # choose which group of datasets to use
    folder_name = "Novice Pointing"
    # folder_name = "Novice Tracing"
    # folder_name = "Surgeon Pointing"
    # folder_name = "Surgeon Tracing"

    # finds the directory
    directory_name = str(pathlib.Path(__file__).parent.resolve()) + "/data/" + folder_name

    [hyperparameters, r2_scores, tremor_r2_scores, rmses, tremor_rmses, training_times, prediction_times] \
        = predict_dir(directory_name, model, prediction_target)

    # prints the average metrics for all datasets
    if model == "SVM":
        print("\nHyperparameters of the model [C_max, C_min, epsilon_max, epsilon_min]:", hyperparameters)
    elif model == "Random Forest":
        print("\nHyperparameters of the model [n_estimators_max, n_estimators_min]:", hyperparameters)
    else:
        print("Invalid model type!")
        exit()
    print(
        "Average R2 score of the model:", str(np.mean(r2_scores)) + "%",
        "\nAverage R2 score of the tremor component:", str(np.mean(tremor_r2_scores)) + "%",
        "\nAverage RMS error of the model:", str(np.mean(rmses)) + "mm",
        "\nAverage RMS error of the tremor component:", str(np.mean(tremor_rmses)) + "mm",
        "\nAverage time taken to train:", str(training_times) + "s",
        "\nAverage time taken to make a prediction:", str(np.mean(prediction_times)) + "s"
    )

    # data for plotting bar chart
    labels = ["Overall R2 score", "Tremor component R2 score"]
    # rounded to better display as bar label
    x_axis_r2 = [round(r2_scores[0]), round(tremor_r2_scores[0])]
    y_axis_r2 = [round(r2_scores[1]), round(tremor_r2_scores[1])]
    z_axis_r2 = [round(r2_scores[2]), round(tremor_r2_scores[2])]
    average_r2 = [round(np.mean(r2_scores)), round(np.mean(tremor_r2_scores))]

    fig, axis = plt.subplots()
    # bar chart properties
    bar_width = 0.2
    x_axis = np.arange(len(labels))

    # bars for each result
    bar1 = axis.bar(x_axis - (3 * bar_width / 2), x_axis_r2, width=bar_width, label="X")
    bar2 = axis.bar(x_axis - (bar_width / 2), y_axis_r2, width=bar_width, label="Y")
    bar3 = axis.bar(x_axis + (bar_width / 2), z_axis_r2, width=bar_width, label="Z")
    bar4 = axis.bar(x_axis + (3 * bar_width / 2), average_r2, width=bar_width, label="Avg")
    # displays bar value above the bar
    axis.bar_label(bar1)
    axis.bar_label(bar2)
    axis.bar_label(bar3)
    axis.bar_label(bar4)

    # axis labels + title
    axis.set_xlabel("R2 score metrics")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title(model + " results based on " + folder_name + " datasets", fontweight="bold")
    # setting ticks
    axis.set_xticks(x_axis)
    axis.set_xticklabels(labels)
    # legend
    axis.legend(title="3D axis", loc=9)
    # tick parameters
    axis.tick_params(axis="x", which="both")
    axis.tick_params(axis="y", which="both")

    plt.show()
