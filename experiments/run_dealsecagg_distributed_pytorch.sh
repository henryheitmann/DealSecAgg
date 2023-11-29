#!/usr/bin/env bash


CLIENT_NUM=$1
WORKER_NUM=$2
DISTRIBUTION=$3
ROUND=$4
EPOCH=$5
BATCH_SIZE=$6
LR=$7
DATA_DIR=$8
CLIENT_OPTIMIZER=$9
CI=${10}
QUANTIZATION=${11}
DROPOUT_NUM=${12}
TRAIN_MODEL_ROUNDS=${13}
TOTAL_NUM_DEALERS=${14}
NUM_DEALERS=${15}

PROCESS_NUM=`expr $WORKER_NUM + $TOTAL_NUM_DEALERS + 1`
echo $PROCESS_NUM

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_dealsecagg.py \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --ci $CI \
  --backend "MPI" \
  --quantization $QUANTIZATION \
  --num_dropouts $DROPOUT_NUM \
  --train_model_rounds $TRAIN_MODEL_ROUNDS \
  --total_num_dealers $TOTAL_NUM_DEALERS \
  --num_dealers $NUM_DEALERS
