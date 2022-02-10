# Structured Sparse Ternary Compression (SSTC)
This repository contains the basic implementation for the simulations from the paper:  
**"Structured Sparse Ternary Compression for Convolutional Layers in Cross-Device Federated Learning"**  
by **Alessio Mora, Luca Foschini, and Paolo Bellavista"**  
{alessio.mora, luca.foschini, paolo.bellavista}@unibo.it  
Corresponding author: alessio.mora@unibo.it

## Setting the environment
To run the code in this repository, i.e. to run the main.py file, you just need to have a Python 
virtual environment with TensorFlow Federated installed. 
Here the official guide to install TFF:
[Install TensorFlow Federated](https://www.tensorflow.org/federated/install).  
We tested the code on Ubuntu 20.04, with Python v3.8.10 and tensorflow-federated v0.19.

## Running the simulation
Activate your virtual environment and run the following command from the sstc directory. It will use 
the default congifuration (1000 total round, 50 clients, 5 local epochs, 0.1 client lr (SGD), 1.0 as server lr (SGD), 16 as batch size, 128 as test batch size,
evaluation on each round).

`python emnist_fedavg_main.py`   

To see what are the input parameters you can pass to the above python script or to know what
is the current default tuning run:

`python emnist_fedavg_main.py --helpshort`

For example, to specify the total number of rounds for the simulation:
`python emnist_fedavg_main.py --total_rounds=50`

### Default Configuration
The default configuration considers 50 clients (randomly selected), 1000 total rounds,
SGD with 0.1 learning rate as local optimizer, 5 local epochs per client.
The default configuration for STC has sparsity = 1%, and fraction of filters = 12.5% for SSTC.
