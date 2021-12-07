#!/bin/bash
#SBATCH --gres=gpu:v100l:2      # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G      # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:30
#SBATCH --output=/project/def-karray/yafathi/MINE/sbatch_output/Eval_MINE_%N-%j.out
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_BLOCKING_WAIT=0  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
echo "r$SLURM_NODEID master: $MASTER_ADDR"

module load python/3.8
source /project/def-karray/yafathi/MINE_ENV/bin/activate
nvidia-smi
sh start_training.sh MASTER_ADDR=$(hostname) MASTER_PORT=1234 N_NODES=1 GPUS_PER_NODE=2 NODE_RANK=0 WORKSPACE=/project/def-karray/yafathi/MINE/ DATASET=kitti_raw VERSION=experiments EXTRA_CONFIG='{"training.gpus": "0,1"}'


