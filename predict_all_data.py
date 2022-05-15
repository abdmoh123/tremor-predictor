# libraries imported
import os
from predict_folder import predict_dir


def predict_dirs(model_type):
    directory_name = "C:/Users/Abdul/OneDrive - Newcastle University/Stage 3/Obsidian Vault/" \
                     "EEE3095-7 Individual Project and Dissertation/Tremor ML/data/"  # desktop
    # directory_name = "C:/Users/abdha/OneDrive - Newcastle University/Stage 3/Obsidian Vault/" \
    #                  "EEE3095-7 Individual Project and Dissertation/Tremor ML/data/"  # laptop
    # list holding all folders to use
    folder_names = [
        "Novice Pointing",
        "Novice Pointing",
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

    # runs the prediction code for each folder
    for folder_name in folder_names:
        predict_dir(directory_name + folder_name, model_type)


if __name__ == '__main__':
    # model = "SVM"
    model = "Random Forest"

    predict_dirs(model)
