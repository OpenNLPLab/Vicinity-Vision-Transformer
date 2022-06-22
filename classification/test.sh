#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

CONFIG=$1
GPUS=$2
PORT=$3
N_TASK_PER_NODE=1
batch_size=20
OUTPUT_DIR=checkpoints/test/
DATA_PATH=/mnt/lustre/share_data/qinzhen/images/images/

# slurm
# spring.submit arun --gpu -n$GPUS --cpus-per-task 1 --ntasks-per-node $N_TASK_PER_NODE -p M3T --quotatype=auto \
# "
# python main.py --config $CONFIG  --data-set CIFAR --batch-size $batch_size --dist-eval --data-path $DATA_PATH \
# --output_dir $OUTPUT_DIR --port $PORT \
# "

# local multi/single GPU
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     --use_env main.py --config $CONFIG  --data-set CIFAR --batch-size $batch_size --output_dir $OUTPUT_DIR \