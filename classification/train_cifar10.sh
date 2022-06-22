#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

CONFIG=$2
GPUS=$1
PORT=${PORT:-8889}
batch_size=$3
OUTPUT_DIR=$4

# spring.submit arun --gres=gpu:$GPUS -n1 --ntasks-per-node=$GPUS --gpu -p MMG \
# "
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     --use_env main.py --config $CONFIG  --data-set CIFAR --batch-size $batch_size --output_dir $OUTPUT_DIR \
# "

CONFIG=configs/pvc_v2/pvc_v2_b1.py
# CONFIG=configs/pvc/pvc_b0.py
CONFIG=configs/vvt/vvt_test.py
# CONFIG=configs/vvt/vvt_tiny.py
# CONFIG=configs/vvt/vvt_small.py
# CONFIG=configs/vvt/vvt_medium.py
# CONFIG=configs/vvt/vvt_large.py
# CONFIG=configs/pvc_v2/pvc_v2_b5.py
GPUS=1
batch_size=2
OUTPUT_DIR=./checkpoints/pvc_b0/

# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     --use_env main.py --config $CONFIG  --data-set CIFAR --batch-size $batch_size --output_dir $OUTPUT_DIR \

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main.py --config $CONFIG  --data-set CIFAR --batch-size $batch_size --output_dir $OUTPUT_DIR \

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     --use_env main.py --config $CONFIG  --data-set CIFAR --batch-size $batch_size --output_dir $OUTPUT_DIR \

# echo 1
# python main.py --config $CONFIG  --data-set CIFAR --batch-size $batch_size --output_dir $OUTPUT_DIR