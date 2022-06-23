export NCCL_LL_THRESHOLD=0

CONFIG=$1
GPUS=$2
PORT=${PORT:-99998}
N_TASK_PER_NODE=$(($2<8?$2:8))
batch_size=20
OUTPUT_DIR=checkpoints/test/
DATA_PATH=/your/data/path/


spring.submit arun --gpu -n$GPUS --cpus-per-task 4 --ntasks-per-node $N_TASK_PER_NODE -p MMG --quotatype=auto \
"
python main.py --config $CONFIG  --data-set IMNET --batch-size $batch_size --dist-eval --data-path $DATA_PATH \
--output_dir $OUTPUT_DIR --port $PORT \
"

# local multi/single GPU
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     --use_env main.py --config $CONFIG  --data-set IMNET --batch-size $batch_size --output_dir $OUTPUT_DIR \


