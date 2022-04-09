# libraries imported
import numpy as np
from scipy import signal
from datetime import datetime


# functions that apply to both simulated and real tremor
import functions.feature_handler as fh
import functions.data_handler as dh
import functions.evaluator as eva
import functions.optimiser as op
import functions.plotter as plt

np.set_printoptions(threshold=50)  # shortens long arrays in the console window


def main():
    file_name = "./data/real_tremor_data.csv"
    time_period = 1 / 250  # a sample is recorded every 0.004 seconds
    no_samples = 10

    # reads data into memory and filters it
    data = dh.read_data(file_name, 200, 4000)  # real tremor data (t, x, y, z, grip force)
    filtered_data = filter_data(data)  # filters data to get an estimate of intended movement (label)
    time = np.array(data[0], dtype='f') * time_period  # samples are measured at a rate of 250Hz

    motion_buffer = []
    label_buffer = []
    time_delay = []
    for i in range(len(data[0])):
        start_time = datetime.now()

        current_motion = [
            data[1][i],
            data[2][i],
            data[3][i]
        ]
        current_label = filter_data(current_motion)
        motion_buffer = add_to_buffer(current_motion, motion_buffer, no_samples)
        label_buffer = add_to_buffer(current_label, label_buffer, no_samples)

        if len(motion_buffer) == no_samples:
            # calculates the features in a separate function
            [features, horizon] = fh.gen_features(time, motion_buffer, label_buffer)

        end_time = datetime.now()
        # measures time taken for each iteration
        iteration_time = (end_time - start_time).total_seconds()
        time_delay.append(iteration_time)
        # ensures that every iteration 'waits' for the next sample to be streamed
        if iteration_time < time_period:
            time.sleep(time_period - iteration_time)


# filters the input data to estimate the intended movement
def filter_data(data):
    time_period = 1 / 250
    nyquist = 1 / (2 * time_period)
    cut_off = 5 / nyquist

    # zero phase filter is used to generate the labels (slow but very accurate)
    [b, a] = signal.butter(2, cut_off, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


# simulates a buffer with motion data being streamed in
def add_to_buffer(motion, buffer, max_length):
    buffer.append(motion)
    # prevents the buffer from exceeding the maximum length
    if len(buffer) > max_length:
        buffer = buffer[len(buffer) - max_length:]
    return buffer


if __name__ == '__main__':
    main()
