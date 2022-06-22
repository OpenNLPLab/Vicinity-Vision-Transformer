#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

CONFIG=$2
GPUS=$1
PORT=${PORT:-8889}
batch_size=$3
OUTPUT_DIR=$4

spring.submit arun --gres=gpu:$GPUS -n1 --ntasks-per-node=$GPUS --gpu -p \
"
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main.py --config $CONFIG  --data-set IMNET --batch-size $batch_size --output_dir $OUTPUT_DIR \
"