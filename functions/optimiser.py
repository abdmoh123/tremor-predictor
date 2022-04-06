import numpy as np
from sklearn import svm
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

import functions.feature_handler as fh
import functions.evaluator as eva


# finds the optimal regularisation parameter (C) and gamma for an SVM regression model
def tune_model(features, labels, parameters=None):
    if parameters is None:
        # hyperparameter choices are tested to find the best combination
        parameters = {
            'kernel': ['rbf'],
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 10],
            'epsilon': [0.01, 0.1, 1, 10, 100]
        }
        # HalvingGridSearch is used instead of GridSearch to speed up the tuning process
        regression = HalvingGridSearchCV(svm.SVR(), parameters)  # SVM regression
    else:
        # model parameters are set based on input
        regression = svm.SVR(
            kernel="rbf",
            C=parameters[0],
            epsilon=parameters[1],
            gamma=parameters[2]
        )
    regression.fit(features, labels)  # fit based on R^2 metric (default)
    return regression


# finds the parameter value that generates the optimal feature values for an SVM regression model
def optimise_parameter(features, labels, parameter):
    rms_error = 100  # initialised as a large value
    feature = []

    if parameter == "horizon":
        final_parameter = 1  # horizon value (starts at 1 to prevent division by zero)
        max_parameter = 50  # limit for the horizon loop
        parameter_increment = 2
        temp_parameter = final_parameter  # temp value for iteration

        # loop puts all possible average features in a list
        while temp_parameter <= max_parameter:
            # calculates the average motion
            feature.append(fh.normalise(fh.calc_average(features[0], temp_parameter)))
            temp_parameter += parameter_increment  # horizon values are incremented in values of 2
    elif parameter == "shift":
        final_parameter = 1  # shift value (when optimising past motion feature)
        max_parameter = 10  # 10 features was optimal for Kabita's implementation
        parameter_increment = 1
        temp_parameter = final_parameter  # temp value for iteration

        # loop puts all possible shifted/past motion features in a list
        while temp_parameter <= max_parameter:
            # shifts the input motion by a set amount to get the past motion as a feature
            feature.append(fh.shift(features[0], temp_parameter))
            temp_parameter += parameter_increment  # increments the shift value

    # loop evaluates models repeatedly to find optimal horizon/shift value
    for i in range(len(feature)):
        # puts the features (including average/shifted) in an array ready for SVR fitting
        temp_features = list(features)
        temp_features.append(feature[i])
        temp_features = np.vstack(temp_features).T

        # SVM with rbf kernel
        regression = svm.SVR(kernel="rbf")
        regression.fit(temp_features, labels)
        predictions = regression.predict(temp_features)

        # calculates the normalised RMS error of the model (tremor component)
        [temp_r2, temp_rmse] = eva.calc_tremor_accuracy(features[0], predictions, labels)
        # final parameter value is only updated if the new value gives a more accurate model
        if temp_rmse < rms_error:
            rms_error = temp_rmse
            final_parameter = (i * parameter_increment) + 1  # calculates the parameter value (based on index)
    return final_parameter  # returns optimal horizon/shift value
