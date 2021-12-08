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

## Training on ShapeNet

We download the shapenet dataset from the following link 
https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip

### DataLoader
Similar to KITTI Raw, we create our own dataloader found in "input_pipelines/shapenet/nerf_dataset.py". We follow the same approach of pixelNeRF in loading the data and apply the same preprocessing steps for a fair comparison. Our main contributions in this script are:

 - Loading the source and image way in a way that fits MINE (since pixelNeRF deals with image pixels in the forms of rays, while MINE deals with the images as a whole). In \_\_init\_\_() for each view in an object ina category, we randomly sample another view of the same object to act as the target view for supervising MINE.
 - Apply the same preporcessing on camera intrinsics and extrinsics as pixelNERF in ( \_\_getitem\_\_() )
### Training Pipeline
 -   Added the code for loading shapenet in "train.py"
 -  Run the training pipeline by running "sbatch_start_train_shapenet.sh"
## Verify Effect of Continuous Depth \& Volumetric Rendering
We used the original code published by MINE to train on llff however did the following changes for each experiment:
 - To activate fixed disparity instead of stratified sampling: set "mpi.fix_disparity=true" in "params_llff.yaml".
 - To activate alpha compositing instead of volumetric rendering: set "mpi.use_alpha=true" in "params_llff.yaml".

## Generalization on LLFF
Uncomment lines 63-69 added to "input_pipelines/llff/nerf_dataset.py".
 - To train on all scenes except "flowers": Uncomment line 45
 - To train on all scenes except "fortress": Uncomment line 46
 - Begin training by running "sbatch_start_train_llff.sh"
## Generalization on KITTI Raw
Edited the main() function in "image_to_video.py" to load the needed sequences only and randomly sample images from each directory and synthesize a video that shows novel views of the input image. 

This experiment uses the pretrained weights published by the authors for KITTI Raw (32 planes).

To run this experiment use "sbatch_synthesize_video.sh"

## Improvement Idea: Multi-View MINE

This improvement idea is considered the largest portion of our code contribution in this project as it required diving deep into the architecture to fit it with multiple source images instead of one. We will go through the areas of the code the are related to this architecture.
### Dataloader
We created a new custom multi view dataloader for the LLFF dataset in "input_pipelines/llff/nerf_dataset_mv.py". This dataloader does the following:

 - We generate a sample for each view present in each of our 8 scenes in the dataset. On average, each scene contains 20-50 views. 
 - We sort the views that are present in a certain scene by their sequence number.
 - For each view of a scene, we randomly sample N+1 views that lie after this view in the sorted list (views that have higher sequence number) to act as source views. We then randomly assign one of the N+1 views as the target view. This assures us that there would be no duplicate source/target view pairs seen during training.
 - Same preporcessing steps done for the LLFF dataset.
 - Adjusted the collate function to batch the source views in a proper way to be loaded in the network
### Training Pipeline
 - List item

Run th
