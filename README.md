# DealSecAgg
This is the official implementation of the paper **DealSecAgg: Efficient Dealer-Assisted Secure Aggregation for Federated Learning**.
It uses an implementation of the [FedML](https://github.com/FedML-AI/FedML) framework for federated learning.

# Getting Started

To be able to run DealSecAgg, you need to have a working MPI like [openMPI](https://www.open-mpi.org/software/ompi/v2.0) installed.
To install necessary dependencies we use the package manager *pip*. All dependencies are defined in the requirements.txt
```
pip install -r requirements.txt
```

The experiments can be run using the `run_dealsecagg_distributed_pytorch.sh` script.
An example with 4 clients is given in the `run.sh` script.

# Reference
WIP