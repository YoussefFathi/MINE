# MINE: Continuous-Depth MPI with Neural Radiance Fields
 
***2021 ML Reproducibility Challenge Code Submission***
This repository contains our reproduibility code for MINE paper. The code portions corresponding to each experiment are mentioned below.

## Slurm Bash Scripts

All of our experiments were implemented on a cloud compute cluster using Slurm which requires submitting jobs to run the experiments using specified bash scripts. You will find such scripts starting with "sbatch" in our root folder.

## Reproducibility on KITTI Raw

### Downloading the dataset
We downloaded the dataset using the script "raw_data_downloader.sh". This script should be placed in the directory where the data will be placed an ran from there.

### Dataloader
The original MINE paper didn't include code for the dataloader of KITTI Raw dataset, so we had to create one ourselves following the class structure defined in the given sample dataloader published for LLFF dataset in "input_pipelines/llff/nerf_dataset.py". Our data loader could be found in "input_pipelines/kitti_raw/nerf_dataset.py". To create the loader we followed the steps made by Tulisiani et al. in their code here (add link), which includes the following steps:

 - Loading the correct training and test sequences for fair comparison (init_img_names_seq_list() function)
 - Preprocessing the given camera intrinsics and extrinsics for both the source and target views. ( read_calib_file() and get_data() )
 - Transformation on the images (__init_img_transforms() )
 - Getting a sample source and target view ( \_\_getitem\_\_(index) )

###  Training Pipeline
The original code included the training pipeline for LLFF dataset only in "train.py" and "synthesis_task.py". Since the pipeline for KITTI Raw is slightly different, we created our modified training code in "synthesis_task_kitti.py" which includes the following changes:

 - Remove the sparse disparity loss from the final loss computation in (loss_fcn_per_scale)
 - Remove any usage of 3D point clouds in the code since they're not loaded from the beginning in the dataloader unlike LLFF.

We also adjusted "train.py" to use our new "synthesis_task_kitti.py" and to include kitti raw dataset calls in "get_dataset()"

The final training is run using batch script "sbatch_start_training.sh" which runs the training on 4 V100 GPUs for the intended time"

The output of the training is printed in "/sbatch_output" directory and we use the final validation step after the final epoch done and report the results as the original code does.

