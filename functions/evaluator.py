import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


# calculates the RMS error of the overall model
def calc_accuracy(actual_output, predictions):
    rms_error = mean_squared_error(actual_output, predictions, squared=False)  # calculates the RMS error
    r2 = 100 * r2_score(actual_output, predictions)  # calculates the r^2 score (percentage accuracy)
    return r2, rms_error


# calculates the R2 score and RMS error of the tremor component of the model
def calc_tremor_accuracy(input_motion, predictions, voluntary_motion):
    # gets the tremor component by subtracting from the voluntary motion
    actual_tremor = np.subtract(input_motion, voluntary_motion)
    predicted_tremor = np.subtract(input_motion, predictions)
    # calculates and returns the normalised RMSE of the tremor component
    return calc_accuracy(predicted_tremor, actual_tremor)
