

> **Project Title**: *Uncertainty-Aware Hybrid Rendering with Gaussian Splatting and NeRF for High-Fidelity Synthesis*  
> **Author**: Patrick Chen  
> **Course**: 16-825 Learning for 3D Vision, Carnegie Mellon University  
> 📄 [View Project Poster (PDF)](./doc/CMU_16825_Final_Project_Poster.pdf)


This project implements a hybrid rendering system that combines the speed of **3D Gaussian Splatting (GS)** and the high fidelity of **NeRF**, guided by a per-pixel uncertainty map to selectively render high-uncertainty regions using NeRF.

---

## Dataset Setup

Download the NeRF Materials dataset from HuggingFace and place it under `./data/materials`.

```bash
sudo apt install git-lfs
git lfs install

git clone https://huggingface.co/datasets/learning3dvision/nerf_materials
cd nerf_materials
unzip materials.zip -d ./data/materials
```

---

## Environment Setup

Clone the repository and install dependencies.

```bash
git clone <this_repo_url>
cd <repo_dir>

# Option 1: pip
pip install -r requirements.txt

# Option 2: conda
conda env create -f environment.yml
conda activate final_proj_venv
```

---

## Training Commands

### NeRF Training

```bash
CUDA_VISIBLE_DEVICES=2 python -m train_nerf.volume_rendering_main \
    --config-name=nerf_materials_highres
```

### Gaussian Splatting (GS) Training

```bash
CUDA_VISIBLE_DEVICES=4 python train_gs.py \
    --data_path ./data/materials/ \
    --img_size 128 \
    --num_itrs 5001 \
    --viz_freq 100 \
    --init_random_numpoints 15000 \
    --use_sched True \
    --use_ssim True \
    --save_freq 500 \
    --val_step 250 \
    --out_path ./output_gnd_256_5001_15000_0.1_0.1_0.04_0.0140_0.009
```

### Uncertainty Map Training (MLP)

```bash
CUDA_VISIBLE_DEVICES=4 python train_uncertainty_map.py \
    --data_path ./data/materials \
    --out_path ./output_uncertainty_lr4e-3_minlr1e-8_numepochs51_warmupsteps500_mlp \
    --img_size 128 \
    --init_random_numpoints 15000 \
    --num_epochs 101 \
    --lr 4e-3 \
    --min_lr 1e-8 \
    --resume_gs \
    --load_gs_ckpt ./output_gnd_128_5001_15000_0.1_0.1_0.04_0.0140_0.009/checkpoint_iter_005000.pth \
    --warmup_steps 500 \
    --val_step 500 \
    --save_step 500
```

### Uncertainty Map Training (UNet)

```bash
CUDA_VISIBLE_DEVICES=2 python train_uncertainty_map.py \
    --data_path ./data/materials \
    --out_path ./output_uncertainty_lr1e-3_minlr6e-9_numepochs201_warmupsteps500_tunet \
    --img_size 128 \
    --init_random_numpoints 15000 \
    --num_epochs 201 \
    --lr 1e-3 \
    --min_lr 6e-9 \
    --resume_gs \
    --load_gs_ckpt ./output_gnd_128_5001_15000_0.1_0.1_0.04_0.0140_0.009/checkpoint_iter_005000.pth \
    --warmup_steps 500 \
    --val_step 500 \
    --save_step 500 \
    --use_tiny_unet
```

---

## Inference/Evaluation

```bash
bash run_hybrid_render.sh
```

---

## Poster

We also provide a project summary poster in the `docs/` folder:

> 📄 [Project Poster (PDF)](./doc/CMU_16825_Final_Project_Poster.pdf)

---

