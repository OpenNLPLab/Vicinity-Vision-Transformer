export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

python run_with_submitit.py --ngpus 8 --nodes 1 --partition MMG --comment spring-submit --model pvt_huge_v2 --batch-size 32 --use-mcloader --output_dir checkpoints/pvt_huge_v2_2n/ --config configs/pvc/pvc_b5.py