Mini-Project 1a: Adaptive Cruise Control in CARLA
=================================================

## Overview

In this homework, we will design an adaptive cruise control system. A car (ego car or
host car) equipped with ACC typically has a sensor, such as radar, that estimates the
distance to the vehicle in front of the same lane (lead car). 

The basic operation of ACC is as follows: the driver sets a cruising speed reference
`vref_host` that is assumed to be constant across this experiment. If the distance
between the host car and the lead car is greater than `dSafe`, then the host car
attempts to travel at the driver-set cruising speed. If the distance between the host
car and the lead car is less than `dSafe`, then the host car tries to maintain at least
the safe distance `dSafe` from the lead car.

The inputs to the ACC are the driver-set reference for the host car, the estimated
distance between the lead car and the host car, and the host car's velocity. The output
of the ACC-based controller the acceleration value for the host car.

## Task

You are given a project with the structure 
```
csci513-miniproject1a                   # root directory
  ├── Carla.Dockerfile                  # Dockerfile to set up the CARLA simulator
  ├── Makefile                          # Makefile to easily run things
  ├── mp1_controller
  │  ├── __init__.py
  │  └── controller.py                  # THE FILE YOU NEED TO EDIT
  ├── mp1_simulator
  │  ├── __init__.py
  │  ├── __main__.py                    # The test runner
  │  ├── misc.py
  │  ├── render.py
  │  └── simulator.py                   # An interface to the simulator
  ├── pyproject.toml
  ├── README.md
  ├── setup.cfg
  ├── setup.py
```

To get started, we have provided you a skeleton of a closed loop system in this package.
The files in `mp1_simulator` interfaces with CARLA to simulate the ego and lead cars.
The primary control logic for the ego car is contained within the `mp1_controller`
directory.

To assist in your control design, the `controller.py` contains a template of a
controller. Your task is to edit the `Controller` class in `controller.py` such that the
`run_step` function takes in an `Observation` (consisting of the speed of the ego
vehicle, the target speed, and the distance to the lead car), and outputs an
acceleration value between `10.0` and `-10.0` (in unit-less control inputs). The
acceleration control input is internally used to set the throttle and breaks for the ego
vehicle.

## Setup

There are two ways to setup the project. The first method uses direct system installation, and the second method uses Docker. 
These instructions assume you have some Linux device. If you are using Windows, check out [Windows Subsystem for
Linux](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).

In all instances you need to have the following prerequisite packages installed:

1. Python >= 3.7
2. GNU Make


### The direct way

You will need to install [CARLA][CARLA] (version 0.9.12) and its [additional
assets](https://carla.readthedocs.io/en/0.9.12/start_quickstart/#import-additional-assets).
Follow the [instructions][inst] in the linked page for your specific platform.

*IMPORTANT:* Remember to verify that you are installing version 0.9.12

[inst]: https://carla.readthedocs.io/en/0.9.12/start_quickstart/
[CARLA]: https://github.com/carla-simulator/carla/blob/master/Docs/download.md


### The Docker way

1. Install [Docker](https://docs.docker.com/get-docker/).
2. If you have a GPU (recommended for CARLA), install
   [`nvidia-docker2`][nvidia-docker2] container runtime to allow `docker` to use GPUs
   during execution.
3. Install GNU Make
4. Run `make build` to build the Docker image. 

  *NOTE:* Docker requires root privileges, so keep an eye out for the `sudo` prompt.

[nvidia-docker2]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide

### Common for both of the above

From the current project directory, run

```shell
$ pip install -e .
```
This command will install the project dependencies, and allow you to run the simulations
easily.

## Running the simulations.

To run the simulations, you will need to have 2 terminals opened.

1. In the first terminal window, start the simulator
  - *Direct way:*
    - Change directories to the directory where you downloaded CARLA
    - Put the folder of csci513-miniproject1 inside folder CARLA/PythonAPI/
    
  - *Docker way:* simply run `make run-carla`.
    - Run
    
```shell
$ ./CarlaUE4.sh 
```

2. In the other terminal, start the controller:
```shell
$ python3 -m mp1_simulator --n-episodes 10
```

You can also run the following to see what options you have for the test runner. 
```shell
$ python3 -m mp1_simulator --help
```

When you run the script, for every episode run by the script, it will save a CSV file in
the `log/` directory. These CSV files (or _traces_) are a recording of the simulation
consisting of only the state variables we deem necessary for evaluating your design.

## Evaluating your design

In this assignment, we will use Signal Temporal Logic (STL) requirements to ensure the
correctness of the controller. To do this, we use the [RT-AMT][rtamt] package (installed
automatically) to define offline monitors for your controller.

[rtamt]: https://github.com/nickovic/

To evaluate the traces you've captured for your controller, simply run the following
command:
```shell
$ python3 -m mp1_evaluation <list of log files>
```
where `<list of log files>` is a placeholder for all the CSV files you need to evaluate.
An example (assuming you are saving the files in `logs/` directory) would be:
```shell
$ python3 -m mp1_evaluation logs/*
```
which will evaluate all the files in that directory.
