# Tremor predictor using SVMs and Random Forest

Implementation by Abdul Hawisa.

Data collected by M<sup>2</sup>S<sup>2</sup> system - see papers referenced below:

- E. L. M. Su, T. L. Win, W. T. Ang, T. C. Lim, C. L. Teo and E. Burdet, "Micromanipulation accuracy in pointing and tracing investigated with a contact-free measurement system", Proc. Annu. Int. Conf. IEEE Eng. Med. Biol. Soc., pp. 3960-3963, Sep. 2009.
- T. L. Win, U. X. Tan, C. Y. Shee and W. T. Ang, "Design and calibration of an optical micro motion sensing system for micromanipulation tasks", Proc. IEEE Int. Conf. Robot. Autom., pp. 3383-3388, Apr. 2007.
- W. T. Latt, U. X. Tan, K. C. Veluvolu, J. K. D. Lin, C. Y. Shee and W. T. Angs, "System to assess accuracy of micromanipulation", Proc. 29th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc., pp. 5743-5746, Aug. 2007.


## Requirements to run code

- Python 3.9 or later versions
- Multiple core CPU

**Libraries to install**

- Numpy
- Scikit-learn
- Scipy
- Matplotlib

## How to setup

1. Download and install python 3.9 or later
    - Make sure add to PATH is ticked
2. Open the terminal/command line
    - Use cmd or Windows Powershell for Windows
    - Use terminal for MacOs and Linux
3. Install the libraries running the command below in the terminal/command line
    - `pip install scikit-learn numpy scipy matplotlib`
4. Configure your IDE to run the scripts (see section below) or just double click to run them

## Scripts to run

- ***offline_predictor.py***:
    - Runs an offline implementation of SVM or Random Forest regression that predicts the voluntary motion
- ***real_time_voluntary_motion.py***:
    - Runs a real-time/online implementation of SVM or Random Forest regression that predicts the voluntary motion
- ***real_time_tremor_component.py***:
    - Runs a real-time implementation of SVM or Random Forest regression that predicts the tremor component
- ***predict_folder.py***:
    - Runs the real-time algorithms on a specified group/folder of datasets and produces bar chart of average results
- ***predict_all_data.py***:
    - Runs the real-time algorithms all datasets and produces bar chart of average results
- ***display_data.py***:
    - Displays a chosen dataset including zero-phase and real-time IIR filters on a line graph
- ***simulated_tremor_SVM.py* (depreciated)**:
    - Runs an offline SVM model on simulated data **(doesn't work anymore)**
