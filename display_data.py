import numpy as np
import functions.plotter as pltr
import functions.data_handler as dh


def display_data(input_data):
    TIME_PERIOD = 1 / 250
    time = np.multiply(input_data[0], TIME_PERIOD)
    zero_phase_filter = [
        dh.filter_data(input_data[1], TIME_PERIOD)[round(0.1 * len(input_data[1])):],
        dh.filter_data(input_data[2], TIME_PERIOD)[round(0.1 * len(input_data[1])):],
        dh.filter_data(input_data[3], TIME_PERIOD)[round(0.1 * len(input_data[1])):]
    ]
    butterworth_filter = [
        dh.filter_data(input_data[1], TIME_PERIOD, False)[round(0.1 * len(input_data[1])):],
        dh.filter_data(input_data[2], TIME_PERIOD, False)[round(0.1 * len(input_data[1])):],
        dh.filter_data(input_data[3], TIME_PERIOD, False)[round(0.1 * len(input_data[1])):]
    ]
    data_to_plot = [
        [input_data[1][round(0.1 * len(input_data[1])):], zero_phase_filter[0], butterworth_filter[0], "X motion (mm)"],
        [input_data[2][round(0.1 * len(input_data[1])):], zero_phase_filter[1], butterworth_filter[1], "X motion (mm)"],
        [input_data[3][round(0.1 * len(input_data[1])):], zero_phase_filter[2], butterworth_filter[2], "X motion (mm)"],
    ]
    axes_labels = ["Original signal", "Zero-phase filter", "IIR Butterworth filter"]
    graph_title = "Graph comparing whole motion with an IIR filter and a Zero-phase filter"
    pltr.plot_model(time[round(0.1 * len(input_data[1])):], data_to_plot, axes_labels, graph_title)


if __name__ == '__main__':
    file_name = "./data/real_tremor_data.csv"
    data = dh.read_data(file_name, 200, 4000)
    display_data(data)
