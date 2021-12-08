# MINE: Continuous-Depth MPI with Neural Radiance Fields
 
***2021 ML Reproducibility Challenge Code Submission***
This repository contains our reproduibility code for MINE paper. The code portions corresponding to each experiment are mentioned below.

## Slurm Bash Scripts

All of our experiments were implemented on a cloud compute cluster using Slurm which requires submitting jobs to run the experiments using specified bash scripts. You will find such scripts starting with "sbatch" in our root folder.

## Reproducibility on KITTI Raw

### Downloading the dataset
We downloaded the dataset using the script "raw_data_downloader.sh". This script should be placed in the directory where the data will be placed and ran from there.

### Dataloader
The original MINE paper didn't include code for the dataloader of KITTI Raw dataset, so we had to create one ourselves following the class structure defined in the given sample dataloader published for LLFF dataset in "input_pipelines/llff/nerf_dataset.py". Our data loader could be found in "input_pipelines/kitti_raw/nerf_dataset.py". To create the loader we followed the steps made by Tulsiani et al. in their code [here](https://github.com/google/layered-scene-inference/blob/59b5d37022f6aaab30dfd4ddcf560923eaf38578/lsi/data/kitti/data.py#L161), which includes the following steps:

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

 - Loading the source and image way in a way that fits MINE (since pixelNeRF deals with image pixels in the forms of rays, while MINE deals with the images as a whole). In \_\_init\_\_() for each view in an object in a category, we randomly sample another view of the same object to act as the target view for supervising MINE.
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
 - Same preprocessing steps done for the LLFF dataset.
 - Adjusted the collate function to batch the source views in a proper way to be loaded in the network
### Training Pipeline
 - Created "synthesis_task_mv.py" to include the updated architecture of multi-view MINE.
	 - Changed init_data() and set_data() to fit with the new dimensionality of source views.
	 - The function network_forward() is responsible for generating the source view MPIs. We allow the network to iteratively pass each source view image through the encoder-decoder models to produce the per-view MPI using the mpi_predictor() function. We stack all the source MPIs in a list.
	 - The estimated MPIs are returned to the function loss_fcn() which is responsible for calculating the loss of the network for each scale using loss_fcn_per_scale().
	 - In loss_fcn_per_scale():
		 - For each source MPI:
			 - We scale all the source and target images and camera parameters to fit with the current needed scale.
			 - Generate the 3D mesh grid for each source view to be used in volumetric rendering using mpi_rendering.get_src_xyz_from_plane_disparity().
			 - Use mpi_rendering.render() to render the synthesized RGB and Disparity maps for each source view using the predicted MPIs and 3D mesh grid.
			 - Use the 3D sparse point cloud estimated using COLMAP to calculate the scale
		 -  Calculate the sparse disparity loss, source L1 loss, source SSIM loss.
	 - Sum all the losses of the source views and add them to the final loss of the network
	 - For the target view:
		 - Use our new function render_novel_view_from_mv() to do the following:
			 - Given the different source MPIs and their poses along with the target view pose,  we generate 3D mesh grid that defines the warped 3D points of the target view with respect to each source view
			 - Use mpi_rendering.render_tgt_rgb_depth_mv() to:
				 -  Render the new view by applying homography warping on each source view to map it to the target view.
				 -  Fuse the estimated target MPIs using **naive averaging method**. Other fusing methods will be implemented in this part as future work.
				 - Use render() to render the final target rgb image.
		 - Calculate the target view losses and add them to the final loss of the network.

Created "train_mv.py" to load the new dataloader and training pipeline of multi-view MINE.

To run the training use "sbatch_start_multi_view_train_llff .sh"
