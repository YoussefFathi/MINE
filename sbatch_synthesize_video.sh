#!/bin/bash
#SBATCH --gres=gpu:v100l:1      # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G      # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-01:00
#SBATCH --output=/project/def-karray/yafathi/MINE/sbatch_output/Eval_Video_MINE_%N-%j.out
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_BLOCKING_WAIT=0  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
echo "r$SLURM_NODEID master: $MASTER_ADDR"

module load python/3.8
source /project/def-karray/yafathi/MINE_ENV/bin/activate
cd /project/def-karray/yafathi/MINE/
nvidia-smi
python image_to_video.py --checkpoint_path /home/yafathi/projects/def-karray/yafathi/MINE/KITTI_Pretrained/checkpoint.pth --gpus 0 --data_path /home/yafathi/projects/def-karray/yafathi/MINE/input_pipelines/kitti_raw/test_1121.png --output_dir .
