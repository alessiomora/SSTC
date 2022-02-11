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
the default configuration (see below for detail).

`python emnist_fedavg_main.py`   

To see what are the input parameters you can pass to the above python script or to know what
is the current default tuning run:

`python emnist_fedavg_main.py --helpshort`

For example, to specify the total number of rounds for the simulation:  
  
`python emnist_fedavg_main.py --total_rounds=50`

The results of the simulation (e.g., accuracy for each round) will be saved on disk, in a 
`logs` folder that can be visualized with Tensorboard or with custom script.

### Default configuration
The default configuration is:

`emnist_fedavg_main.py:`  
 `  --batch_size: Batch size used on the client.
    (default: '16')
    (an integer)`  
`  --client_epochs_per_round: Number of epochs in the client to take per round.
    (default: '5')
    (an integer)`  
` --client_learning_rate: Client learning rate.
    (default: '0.1')
    (a number)`  
` --rounds_per_eval: How often to evaluate
    (default: '1')
    (an integer)`    
`  --server_learning_rate: Server learning rate.
    (default: '1.0')
    (a number)`  
`  --sstc_filter_fraction: Client learning rate (0, 1]
    (default: '0.125')
    (a number)`  
`  --stc_sparsity: STC sparsity (0, 1]
    (default: '0.01')
    (a number)`  
`  --test_batch_size: Minibatch size of test data.
    (default: '128')
    (an integer)`  
`  --total_rounds: Number of total training rounds.
    (default: '1000')
    (an integer)`  
`  --train_clients_per_round: How many clients to sample per round.
    (default: '50')
    (an integer)`


### Enabling STC or SSTC
To enable STC, use a value for `--stc_sparsity` < 1 and `--sstc_filter_fraction=1.0`.
  
To enable SSTC, use a value for `--stc_sparsity` < 1 and a value for `--sstc_filter_fraction` < 1.0.

For uncompressed FedAvg, use `--stc_sparsity=1.0`.
  
At default, the simulation will run SSTC with `--stc_sparsity=0.01` and `--stc_sparsity=0.125`, which are the
best configuration for the communication-efficiency/model accuracy trade-off (Fig.3 (a), (b) in the paper).