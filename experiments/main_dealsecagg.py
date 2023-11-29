import argparse
import logging
import os
import random
import socket
import sys
import traceback
from time import sleep

import numpy as np
import psutil
import setproctitle
import torch
from mpi4py import MPI

import wandb

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10, \
    dummy_load_partition_data_cifar10
from fedml_api.model.cv.resnet import resnet56

from fedml_api.distributed.dealsecagg.API import FedML_init, FedML_DealSecAgg_distributed

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument("--model", type=str, default="resnet56", metavar="N", help="neural network used in training")
    parser.add_argument("--dataset", type=str, default="cifar10", metavar="N", help="dataset used for training")
    parser.add_argument("--data_dir", type=str, default="./../../../data/cifar10", help="data directory")

    parser.add_argument(
        "--partition_method",
        type=str,
        default="hetero",
        metavar="N",
        help="how to partition the dataset on local workers",
    )

    parser.add_argument(
        "--partition_alpha", type=float, default=0.5, metavar="PA", help="partition alpha (default: 0.5)"
    )

    parser.add_argument(
        "--client_num_in_total", type=int, default=1000, metavar="NN", help="number of workers in a distributed cluster"
    )

    parser.add_argument("--client_num_per_round", type=int, default=4, metavar="NN", help="number of workers")

    parser.add_argument(
        "--batch_size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )

    parser.add_argument("--client_optimizer", type=str, default="adam", help="SGD with momentum; adam")

    parser.add_argument("--backend", type=str, default="MPI", help="Backend for Server and Client")

    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")

    parser.add_argument("--wd", help="weight decay parameter;", type=float, default=0.001)

    parser.add_argument("--epochs", type=int, default=5, metavar="EP", help="how many epochs will be trained locally")

    parser.add_argument("--comm_round", type=int, default=10, help="how many round of communications we shoud use")

    parser.add_argument("--frequency_of_the_test", type=int, default=1, help="the frequency of the algorithms")

    parser.add_argument(
        "--grpc_ipconfig_path",
        type=str,
        default="grpc_ipconfig.csv",
        help="config table containing ipv4 address of grpc server",
    )

    parser.add_argument("--ci", type=int, default=0, help="CI")

    parser.add_argument("--quantization", type=str, default='float32', help="data type to quantize float32 to")

    parser.add_argument("--num_dropouts", type=int, default=0, help="number of dropouts per round")

    parser.add_argument("--train_model_rounds", type=int, default=0, help="the number iterations the model is trained "
                                                                          "in each round")
    parser.add_argument("--total_num_dealers", type=int, default=1, help="total number of dealers available to clients")

    parser.add_argument("--num_dealers", type=int, default=1, help="number of dealers each client uses per round")

    args = parser.parse_args()
    return args

def load_data(args):
    data_loader = load_partition_data_cifar10
    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = data_loader(
        args.dataset,
        args.data_dir,
        args.partition_method,
        args.partition_alpha,
        args.client_num_in_total,
        args.batch_size,
    )
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset


def load_dummy_data(args):
    data_loader = dummy_load_partition_data_cifar10
    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = data_loader(
        args.dataset,
        args.data_dir,
        args.partition_method,
        args.partition_alpha,
        args.client_num_in_total,
        args.batch_size,
    )
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset


def create_model(args, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % ("resnet56", output_dim))
    model = resnet56(class_num=output_dim)
    return model

if __name__ == "__main__":
    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # customize the process name
    str_process_name = "DealSecAgg(distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(
        level=logging.DEBUG,
        format=str(process_id) + " - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )
    hostname = socket.gethostname()
    logging.info(
        "#############process ID = "
        + str(process_id)
        + ", host name = "
        + hostname
        + "########"
        + ", process ID = "
        + str(os.getpid())
        + ", process Name = "
        + str(psutil.Process(os.getpid()))
    )
    logging.info(args)
    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if args.train_model_rounds > 0 and process_id == 0:
        wandb.init(
            # project="federated_dealsecagg",
            project="fedml",
            name="DealSecAgg(d)"
                 + str(args.partition_method)
                 + "r"
                 + str(args.comm_round)
                 + "-e"
                 + str(args.epochs)
                 + "-lr"
                 + str(args.lr),
            config=args,
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = torch.device("cpu")

    # load data
    if args.train_model_rounds == 0:
        dataset = load_dummy_data(args)
    else:
        dataset = load_data(args)
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, output_dim=dataset[7])

    try:
        # start "federated averaging (DealSecAgg)"
        FedML_DealSecAgg_distributed(
            process_id,
            worker_number,
            args.total_num_dealers,
            device,
            comm,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            args,
        )
    except Exception as e:
        logging.info(e)
        logging.info("traceback.format_exc():\n%s" % traceback.format_exc())
        sleep(3)
        MPI.COMM_WORLD.Abort()
