#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

CONFIG=$3
GPUS=$1
PORT=${PORT:-99998}

spring.submit arun --gres=gpu:$GPUS -n 1 --ntasks-per-node=$2 --gpu \
"
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main.py --config $CONFIG  --data-set IMNET \
"