# KalmanNet with Visual Observation

Most files are modified based on the KalmanNet_TSP code https://github.com/KalmanNet/KalmanNet_TSP, adding the suffix of "_visual".

## Running code


There are 3 main files simulating the Auto-Encoder, the seperated encoder & KNet case and the combination case respectively.

* Auto-Encoder

```
python3 main_AE.py
```


* Seperated encoder & KNet case 

```
python3 main_visual.py
```


* "CNN encoder + KNet" combination case 

```
python3 main_combine.py
```


## Introduction to other files

* AE Process/ & saved_models/

These folders are used to store the simulation results of Auto-Encoder (or Encoder, Decoder).


* KNet/

This folder is used to store the trained KalmanNet model and pipeline results of your simulation.

* Simulations/

This folder stores the dataset for synthetic linear case, as well as model and parameter files for pendulum.

* EKF_visual.py & EKF_test_visual.py

This is where we define Extended Kalman Filter (EKF) model and its testing.

* Extended_data_visual.py

This is a parameter-setting and data-generating/loading/preprocessing file.

You could set the number of Training/Cross-Validation(CV)/Testing examples through N_E/N_CV/N_T.

You could set trajectory length of Training/CV examples for synthetic linear case through T, while T_test is for Testing trajectory length.

You could set the synthetic linear model through F10 and H10, and uncomment 2x2, 5x5, 10x10 according to your needs.

* KalmanNet_nn_LinearCase_OldArch_visual.py & KalmanNet_nn_OldArch_visual.py & KalmanNet_nn_NewArch_visual.py

As specified by their names, these files describe the architecture #1 or #2 of KalmanNet model for linear and non-linear cases respectively.

* Linear_sysmdl_visual.py & Extended_sysmdl_visual.py & KalmanNet_sysmdl.py

The first two files are system model files for linear and non-linear cases respectively. They store system information (F/f, H/h, Q, R, m, n) and define functions for generating data according to your system model. While KalmanNet_sysmdl.py is specifically defined for KalmanNet with architecture #2.

* filling_paths.py

This is where you switch between different system models. (can add models other than pendulum in the future)

* Linear_KF_visual.py & KalmanFilter_test_visual.py

This is where we define Linear Kalman Filter (KF) model and its testing.


* PF_test.py & UKF_test.py

These files defines the testing of Particle Filter (PF) and Unscented Kalman Filter (UKF) benchmarks.

* Pipeline_KF_visual.py & Pipeline_combine.py

These are the pipeline files for "KNet only" and "CNN encoder + KNet combination" respectively. The pipeline mainly defines the Training/CV/Testing processes of KalmanNet.

* Plot.py

This file mainly defines the plotting of training process, histogram, MSE results, etc.

* Vanilla_rnn.py

This file defines a baseline comparison for KNet, which is purely RNN implementation.

* visual_supplementary.py

This file mainly defines the fully connected H model and some supplementary parameters/functions for the synthetic linear dataset.






