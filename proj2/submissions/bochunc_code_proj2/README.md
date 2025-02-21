# 16-825 Assignment 2: Single View to 3D

Goals: In this assignment, you will explore the types of loss and decoder functions for regressing to voxels, point clouds, and mesh representation from single view RGB input. 
## Note

Please follow the following steps to install necessary packages:
1. conda create -n {$venv_name} python=3.10 -y
2. conda activate {$venv_name}
3. conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
4. pip install "numpy<2.0.0"
5. pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html
6. python -c "from pytorch3d.ops import knn_points; print('PyTorch3D is working')"
7. python -c "import torch; print(torch.cuda.is_available())"
8. python -c "from torchvision.ops import nms; print('NMS loaded successfully')"
9. python -c "import numpy; print(numpy.__version__)"
10. pip install -r ./requirement.txt

Followng are the commands to run each of the questions:

#Question 1.1
python fit_data.py --type 'vox'

#Question 1.2
python fit_data.py --type 'point' --max_iter 200000

#Question 1.3
python fit_data.py --type 'mesh'

#Question 2.1
#Vox Train
python3 train_model.py --type 'vox' --max_iter 30001 --save_freq 1000 --batch_size 96 --num_workers 14  --lr 1.8e-3  --use_cas 1 --cas_warmup_steps 700 --cas_min_lr 3e-4 --cas_final_lr 8e-5 --use_step_update 1 --output_path ./outputs_max_iter10001_bz96_num_workers_14_lr_1.8e-3_use_cas_1_cas_warmup_steps_700_cas_min_lr_3e-4_cas_final_lr_8e_-5_use_step_update_1_bk

#Vox Eval
If you do not train Question 2.1 using above command:
1. Download weight to ./outputs_max_iter10001_bz96_num_workers_14_lr_1.8e-3_use_cas_1_cas_warmup_steps_700_cas_min_lr_3e-4_cas_final_lr_8e_-5_use_step_update_1_bk from this link: https://drive.google.com/file/d/1LYWtviG2WnZuogf511Lxi2GCSRBBclzP/view?usp=sharing
2. Run:
 python eval_model.py --type 'vox' --load_checkpoint --eval_chk_file ./outputs_max_iter10001_bz96_num_workers_14_lr_1.8e-3_use_cas_1_cas_warmup_steps_700_cas_min_lr_3e-4_cas_final_lr_8e_-5_use_step_update_1_bk/checkpoint_vox.pth --output_eval_path ./outputs_max_iter10001_bz96_num_workers_14_lr_1.8e-3_use_cas_1_cas_warmup_steps_700_cas_min_lr_3e-4_cas_final_lr_8e_-5_use_step_update_1_bk/

#Question 2.2
#Point Train
python train_model.py --type 'point' --save_freq 600 --lr_sch_on 1 --n_point 5000 --max_iter 8001 --lr 9e-5 --batch_size 32 --early_stop_iter 601 --output_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk

#Point Eval
If you do not train using above command:
1. Download weight to ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk from this link: https://drive.google.com/file/d/1NiQj11Asg4fOPiJAQvPUbW-Uir4Ur3rk/view?usp=sharing
2. Run:
python eval_model.py --type 'point' --load_checkpoint --n_points 5000 --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk/checkpoint_point.pth --output_eval_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk

#Question 2.3
#Mesh Train
python ./train_model.py --type 'mesh' --max_iter 4001 --batch_size 64 --num_workers 8 --save_freq 700 --early_stop_iter 701 --lr 9e-5 --lr_sch_on 1 --output_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk

#Mesh Eval
If you do not train using above command:
1. Download weight to ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk from this link: https://drive.google.com/file/d/11P27Z_ZB-4PzYqUDB4YqhIXCE6Jvdypc/view?usp=sharing
2. Run:
python eval_model.py --type 'mesh' --num_workers 8 --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk

#Question 2.5
#Point Cloud Train
python train_model.py --type 'point' --save_freq 600 --lr_sch_on 1 --n_point 1000 --max_iter 8001 --lr 9e-5 --batch_size 32 --early_stop_iter 601 --output_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_1000_bk

python train_model.py --type 'point' --save_freq 600 --lr_sch_on 1 --n_point 2500 --max_iter 8001 --lr 9e-5 --batch_size 32 --early_stop_iter 601 --output_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_2500_bk

python train_model.py --type 'point' --save_freq 600 --lr_sch_on 1 --n_point 5000 --max_iter 8001 --lr 9e-5 --batch_size 32 --early_stop_iter 601 --output_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk

python train_model.py --type 'point' --save_freq 500 --lr_sch_on 1 --n_point 10000 --max_iter 8001 --lr 9e-5 --batch_size 32 --early_stop_iter 501 --output_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_10000_bk

#Point Cloud Eval
If you do not train using above command:
1. Download weight to ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_1000_bk  from this link: https://drive.google.com/file/d/1TEKlOTYivMgft9bsSUl6nBkQduCuSxqi/view?usp=sharing
2. Download weight to ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_2500_bk from this link: https://drive.google.com/file/d/13hv0Mc3qfBHG-9Lb-CgDALTM5UKJhNOM/view?usp=sharing
3. Download weight to ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk from this link: https://drive.google.com/file/d/14-8uO8lxAssA7iJPievQ121g9WgsiKoY/view?usp=sharing
4. Download weight to ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_10000_bk from this link: https://drive.google.com/file/d/1iZREL88lqqf-Pa3y9hf64Z42PReGzRFo/view?usp=sharing
5. Run:
python eval_model.py --type 'point' --load_checkpoint --n_points 1000 --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_1000_bk/checkpoint_point.pth --output_eval_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_1000_bk/

python eval_model.py --type 'point' --load_checkpoint --n_points 2500 --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_2500_bk/checkpoint_point.pth --output_eval_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_2500_bk/

python eval_model.py --type 'point' --load_checkpoint --n_points 5000 --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk/checkpoint_point.pth --output_eval_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk

python eval_model.py --type 'point' --load_checkpoint --n_points 10000 --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_10000_bk/checkpoint_point.pth --output_eval_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_10000_bk


#Mesh Train
python ./train_model.py --type 'mesh' --max_iter 4001 --batch_size 64 --num_workers 8 --save_freq 700 --early_stop_iter 701 --lr 9e-5 --lr_sch_on 1 --w_chamfer 0.5 --output_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_0.5_lr_9e-5_lr_sch_on_1_loss_orig_bk

python ./train_model.py --type 'mesh' --max_iter 4001 --batch_size 64 --num_workers 8 --save_freq 700 --early_stop_iter 701 --lr 9e-5 --lr_sch_on 1 --output_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk

python ./train_model.py --type 'mesh' --max_iter 4001 --batch_size 64 --num_workers 8 --save_freq 700 --early_stop_iter 701 --lr 9e-5 --lr_sch_on 1 --w_chamfer 2 --output_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_2_lr_9e-5_lr_sch_on_1_loss_orig_bk

python ./train_model.py --type 'mesh' --max_iter 4001 --batch_size 64 --num_workers 8 --save_freq 700 --early_stop_iter 701 --lr 9e-5 --lr_sch_on 1 --w_chamfer 5 --output_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_5_lr_9e-5_lr_sch_on_1_loss_orig_bk

#Mesh Eval
If you do not train using above command:
1. Download weight to ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_0.5_lr_9e-5_lr_sch_on_1_loss_orig_bk from this link: https://drive.google.com/file/d/1YZ0sfaeb0qwo-Rs9CWgbE-PZwr9E2wbU/view?usp=sharing
2. Download weight to ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk from this link: https://drive.google.com/file/d/1Is4faMQlKBi4c6FAJd9Ze5cAz7mVjDBv/view?usp=sharing
3. Download weight to ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_2_lr_9e-5_lr_sch_on_1_loss_orig_bk from this link: https://drive.google.com/file/d/1LM9wJfgwMo1Cp9TftoyWwxh5En38h6WS/view?usp=sharing
4. Download weight to ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_5_lr_9e-5_lr_sch_on_1_loss_orig_bk from this link: https://drive.google.com/file/d/1UVNxZf1oI636bgvgS3yTi8KQ0YX7ZfOk/view?usp=sharing
5. Run:
python eval_model.py --type 'mesh' --num_workers 8 --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_0.5_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_0.5_lr_9e-5_lr_sch_on_1_loss_orig_bk

python eval_model.py --type 'mesh' --num_workers 8 --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk

python eval_model.py --type 'mesh' --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_2_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_2_lr_9e-5_lr_sch_on_1_loss_orig_bk

python eval_model.py --type 'mesh'  --num_workers 8 --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_5_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_5_lr_9e-5_lr_sch_on_1_loss_orig_bk


#Question 2.6
python ./vis_activation.py --type 'vox' --load_checkpoint --eval_chk_file ./outputs_max_iter10001_bz96_num_workers_14_lr_1.8e-3_use_cas_1_cas_warmup_steps_700_cas_min_lr_3e-4_cas_final_lr_8e_-5_use_step_update_1_bk/checkpoint_vox.pth --output_eval_path ./vis_saliency_map_vox
python ./vis_activation.py --type 'point' --load_checkpoint --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk/checkpoint_point.pth --output_eval_path ./vis_saliency_map_point
python ./vis_activation.py --type 'mesh' --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./vis_saliency_map_mesh

#Question 3.1
#Train
python ./train_model_occnet.py  --type 'occ' --max_iter 10001 --batch_size 16 --num_workers 14 --save_freq 500 --use_cas 1 --cas_warmup_steps 500 --lr 1e-4 --cas_min_lr 7e-6 --cas_final_lr 8e-7 --cas_T_0 2000 --cas_T_mult 2 --use_step_update 1 --n_sample_pt 256 --load_checkpoint --output_path ./test_chkpt_model4_3_bceloss

#Eval
If you do not train using above command:
1. Download weight to ./test_chkpt_model4_3_bceloss from this link: https://drive.google.com/file/d/1aYj9MEGdVOT1m_J4vnZTwFfGqbOtbL2m/view?usp=sharing
2. Run:
 python ./eval_model_occ.py --type 'occ' --load_checkpoint --eval_chk_file ./test_chkpt_model4_3_bceloss/checkpoint_occ.pth --output_eval_path ./test_chkpt_model4_3_bceloss


## Table of Contents
0. [Setup](#0-setup)
1. [Exploring Loss Functions](#1-exploring-loss-functions)
2. [Reconstructing 3D from single view](#2-reconstructing-3d-from-single-view)
3. [Exploring other architectures / datasets](#3-exploring-other-architectures--datasets-choose-at-least-one-more-than-one-is-extra-credit)
## 0. Setup

Please download and extract the dataset for this assigment. We provide two versions for the dataset, which are hosted on huggingface. 

* [here](https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset) for a single-class dataset which contains one class of chair. Total size 7.3G after unzipping.

Download the dataset using the following commands:

```
$ sudo apt install git-lfs
$ git lfs install
$ git clone https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset
```

* [here](https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset_full) for an extended version which contains three classes, chair, plane, and car.  Total size 48G after unzipping. Download this dataset with the following command:

```
$ git lfs install
$ git clone https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset_full
```

Downloading the datasets may take a few minutes. After unzipping, set the appropriate path references in `dataset_location.py` file [here](dataset_location.py).

The extended version is required for Q3.3; for other parts, using single-class version is sufficient.

Make sure you have installed the packages mentioned in `requirements.txt`.
This assignment will need the GPU version of pytorch.

## 1. Exploring loss functions
This section will involve defining a loss function, for fitting voxels, point clouds and meshes.

### 1.1. Fitting a voxel grid (5 points)
In this subsection, we will define binary cross entropy loss that can help us <b>fit a 3D binary voxel grid</b>.
Define the loss functions `voxel_loss` in [`losses.py`](losses.py) file. 
For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid. 

Visualize the optimized voxel grid along-side the ground truth voxel grid using the tools learnt in previous section.

### 1.2. Fitting a point cloud (5 points)
In this subsection, we will define chamfer loss that can help us <b> fit a 3D point cloud </b>.
Define the loss functions `chamfer_loss` in [`losses.py`](losses.py) file.
<b>We expect you to write your own code for this and not use any pytorch3d utilities. You are allowed to use functions inside pytorch3d.ops.knn such as knn_gather or knn_points</b>

Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud. 

Visualize the optimized point cloud along-side the ground truth point cloud using the tools learnt in previous section.

### 1.3. Fitting a mesh (5 points)
In this subsection, we will define an additional smoothening loss that can help us <b> fit a mesh</b>.
Define the loss functions `smoothness_loss` in [`losses.py`](losses.py) file.

For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh. 

Visualize the optimized mesh along-side the ground truth mesh using the tools learnt in previous section.

## 2. Reconstructing 3D from single view
This section will involve training a single view to 3D pipeline for voxels, point clouds and meshes.
Refer to the `save_freq` argument in `train_model.py` to save the model checkpoint quicker/slower. 

We also provide pretrained ResNet18 features of images to save computation and GPU resources required. Use `--load_feat` argument to use these features during training and evaluation. This should be False by default, and only use this if you are facing issues in getting GPU resources. You can also enable training on a CPU by the `device` argument. Also indiciate in your submission if you had to use this argument. 

### 2.1. Image to voxel grid (20 points)
In this subsection, we will define a neural network to decode binary voxel grids.
Define the decoder network in [`model.py`](model.py) file for `vox` type, then reference your decoder in [`model.py`](model.py) file

Run the file `python train_model.py --type 'vox'`, to train single view to voxel grid pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval_model.py` file using:
`python eval_model.py --type 'vox' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D voxel grid and a render of the ground truth mesh.

### 2.2. Image to point cloud (20 points)
In this subsection, we will define a neural network to decode point clouds.
Similar as above, define the decoder network in [`model.py`](model.py) file for `point` type, then reference your decoder in [`model.py`](model.py) file.

Run the file `python train_model.py --type 'point'`, to train single view to pointcloud pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth point cloud and predicted  point cloud in `eval_model.py` file using:
`python eval_model.py --type 'point' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`.

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D point cloud and a render of the ground truth mesh.


### 2.3. Image to mesh (20 points)
In this subsection, we will define a neural network to decode mesh.
Similar as above, define the decoder network in [`model.py`](model.py) file for `mesh` type, then reference your decoder in [`model.py`](model.py) file.

Run the file `python train_model.py --type 'mesh'`, to train single view to mesh pipeline, feel free to tune the hyperparameters as per your need. We also encourage the student to try different mesh initializations (i.e. replace `ico_sphere` by other shapes).


After trained, visualize the input RGB, ground truth mesh and predicted mesh in `eval_model.py` file using:
`python eval_model.py --type 'mesh' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`.

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted mesh and a render of the ground truth mesh.

### 2.4. Quantitative comparisions(10 points)
Quantitatively compare the F1 score of 3D reconstruction for meshes vs pointcloud vs voxelgrids.
Provide an intutive explaination justifying the comparision.

For evaluating you can run:
`python eval_model.py --type voxel|mesh|point --load_checkpoint`


On your webpage, you should include the f1-score curve at different thresholds for voxelgrid, pointcloud and the mesh network. The plot is saved as `eval_{type}.png`.

### 2.5. Analyse effects of hyperparams variations (10 points)
Analyse the results, by varying a hyperparameter of your choice.
For example `n_points` or `vox_size` or `w_chamfer` or `initial mesh (ico_sphere)` etc.
Try to be unique and conclusive in your analysis.

### 2.6. Interpret your model (15 points)
Simply seeing final predictions and numerical evaluations is not always insightful. Can you create some visualizations that help highlight what your learned model does? Be creative and think of what visualizations would help you gain insights. There is no `right' answer - although reading some papers to get inspiration might give you ideas.


## 3. Exploring other architectures / datasets. (Choose at least one! More than one is extra credit)

### 3.1 Implicit network (10 points)
Implement an implicit decoder that takes in as input 3D locations and outputs the occupancy value. Start with a simple implementation of a network that predicts the occupancy given the image feture and a 3d coordinate as input. You will need to create a meshgrid of 32x32x32 in the normalized coordinate space of (-1,1)^3 to predict the full occupancy output. 

Some papers for inspiration [[1](https://arxiv.org/abs/2003.04618),[2](https://arxiv.org/abs/1812.03828)]

### 3.2 Parametric network (10 points)
Implement a parametric function that takes in as input sampled 2D points and outputs their respective 3D points. 
Some papers for inspiration [[1](https://arxiv.org/abs/1802.05384),[2](https://arxiv.org/abs/1811.10943)]

### 3.3 Extended dataset for training (10 points)
In the extended dataset, we provide a `split_3c.json` file that specifies the train/test split for the extended dataset.

Update `dataset_location.py` so that we train the 3D reconstruction model on an extended dataset containing three classes (chair, car, and plane). Choose at least one of three models (voxel, point cloud, or mesh) to train and evaluate.

After training, compare the quantitative and qualitative results of "training on one class" VS "training on three classes". Explain your thoughts and analysis.

(Hints: for example, given the same testing samples in `chair` class, how does F1 score change comparing "training on one class" and "training on three classes"? How does the 3D consistency / diversity of the output samples change?)
