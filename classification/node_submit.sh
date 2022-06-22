#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0
# export NCCL_IB_DISABLE=1

CONFIG=$2
GPUS=$1
NODES=2
batch_size=$3
PORT=${PORT:-8889}
OUTPUT_DIR=$4
RESUME=$5


if [ -z $RESUME ]
then
    spring.submit arun --gpu -n$GPUS --cpus-per-task 4   --ntasks-per-node 8 -p MMG  \
    "
    python main.py --config $CONFIG  --data-set IMNET --batch-size $batch_size --dist-eval \
    --output_dir $OUTPUT_DIR \
    "
else
    echo $RESUME
    spring.submit arun --gpu -n$GPUS --cpus-per-task 4   --ntasks-per-node 8 -p MMG \
    "
    python main.py --config $CONFIG  --data-set IMNET --batch-size $batch_size --dist-eval \
    --output_dir $OUTPUT_DIR --resume $RESUME \
    "
fi




# spring.submit arun --gpu -n$GPUS --cpus-per-task 4   --ntasks-per-node $GPUS -p MMG  \
# "
# python main.py --config $CONFIG  --data-set IMNET --batch-size $batch_size --dist-eval --input-size 224 \
# "

#  --nproc_per_node=8

# spring.submit arun --gpu -n$GPUS --cpus-per-task 5   --ntasks-per-node 8 -p MMG  \
# "
# python main.py --config $CONFIG  --data-set IMNET --batch-size $batch_size --dist-eval \
# --resume $RESUME 
# "

