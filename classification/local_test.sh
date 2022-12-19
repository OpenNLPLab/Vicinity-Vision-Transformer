export NCCL_LL_THRESHOLD=0

GPUS=1
PORT=${PORT:-8889}
batch_size=2

echo $CONFIG

PROG=main.py
CONFIG_DIR=configs/vvt_abl
DATA=~/Desktop/pvc/data

for ARCH in vvt_gcnet_tiny vvt_no_conv_tiny vvt_softmax_tiny
do
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        --use_env $PROG --data-set CIFAR --batch-size $batch_size --num_workers 1 --lr 3e-3 \
        --data-path $DATA \
        --config ${CONFIG_DIR}/${ARCH}.py \
        --fp32-resume \
        --test \
        --num_workers 0 \
        --warmup-epochs 10 \
        2>&1 | tee log/${ARCH}.log
done

        