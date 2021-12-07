#!/bin/bash
#SBATCH --gres=gpu:v100l:1      # Request GPU "generic resources"
#SBATCH --cpus-per-task=2  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G      # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=1-00:00
#SBATCH --output=/project/def-karray/yafathi/MINE/sbatch_output/MINE_tensor_%N-%j.out
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.8
source /project/def-karray/yafathi/MINE_ENV/bin/activate
tensorboard --logdir=/project/def-karray/yafathi/MINE/experiments --host 0.0.0.0 
# nvidia-smi
# sh start_training.sh MASTER_ADDR="localhost" MASTER_PORT=1234 N_NODES=1 GPUS_PER_NODE=4 NODE_RANK=0 WORKSPACE=/project/def-karray/yafathi/MINE/ DATASET=kitti_raw VERSION=debug EXTRA_CONFIG='{"training.gpus": "0,1,2,3"}'


