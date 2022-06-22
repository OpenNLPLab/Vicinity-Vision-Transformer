#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

CONFIG=$2
GPUS=$1
PORT=${PORT:-8888}

spring.submit arun --gres=gpu:$GPUS -n2 --ntasks-per-node=$GPUS --gpu \
"
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main.py --config $CONFIG  --data-set IMNET \
"