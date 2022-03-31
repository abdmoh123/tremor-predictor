import numpy as np
from sklearn.metrics import mean_squared_error


# calculates the normalised RMS error of the overall model
def calc_accuracy(actual_output, predictions):
    rms_error = mean_squared_error(actual_output, predictions, squared=False)  # calculates the RMS error
    nrms_error = rms_error / np.std(actual_output)  # normalises the RMSE using the standard deviation
    return nrms_error


# calculates the normalised RMS of the tremor component of the model
def calc_tremor_accuracy(input_motion, predictions, voluntary_motion):
    # gets the tremor component by subtracting from the voluntary motion
    actual_tremor = np.subtract(input_motion, voluntary_motion)
    predicted_tremor = np.subtract(input_motion, predictions)
    # calculates and returns the normalised RMSE of the tremor component
    return calc_accuracy(predicted_tremor, actual_tremor)
