#!/bin/bash

# Set fixed paths
CONFIG_PATH="./train_nerf/configs/nerf_materials_highres.hybrid.yaml"
#CONFIG_PATH="./train_nerf/configs/nerf_materials_highres.yaml"
GS_CKPT="./output_gnd_128_5001_15000_0.1_0.1_0.04_0.0140_0.009/checkpoint_iter_005000.pth"
UNCERTAINTY_CKPT="./output_uncertainty_lr4e-3_minlr1e-8_numepochs51_warmupsteps500_mlp/uncertainty_mlp.10100.pth"
#UNCERTAINTY_CKPT="./output_uncertainty_lr1e-3_minlr6e-9_numepochs201_warmupsteps500_tunet/uncertainty_tunet.20100.pth"
VAL_INDEX=0
INSPECT_IDS="0 20 40 60 80 99"
#INSPECT_IDS="0 40 80"

# Loop over thresholds
for THRESHOLD in 0.05 0.1 0.2 0.3 0.4 0.5; do
    # Set output path based on threshold
    OUT_PATH="./output_hybrid_threshold${THRESHOLD}_renderall_init15000_simple-mlp4"
    #OUT_PATH="./output_hybrid_threshold${THRESHOLD}_renderall_init15000_tunet4"
    #OUT_PATH="./output_hybrid_threshold${THRESHOLD}_renderall_init15000_rerun"
    #OUT_PATH="./output_hybrid_threshold${THRESHOLD}_renderall_init15000_coarse-fine-retrain"

    # Run command
    CUDA_VISIBLE_DEVICES=1 python ./hybrid_render.py \
        --config_path "$CONFIG_PATH" \
        --gs_ckpt "$GS_CKPT" \
        --uncertainty_ckpt "$UNCERTAINTY_CKPT" \
        --threshold "$THRESHOLD" \
        --val_index "$VAL_INDEX" \
        --out_path "$OUT_PATH" \
        --render_all \
        --inspect_ids $INSPECT_IDS \
        > ./log
        #--use_tiny_unet

    # Move log into output directory
    mv ./log "$OUT_PATH"/log
    #echo "new re-rerun" >> "$OUT_PATH"/log
    #echo "new re-rerun w/ coarse-fine-retrianing" >> "$OUT_PATH"/log
    #echo "new re-rerun w/ tiny unet" >> "$OUT_PATH"/log
    #echo "new re-rerun w/ simple-mlp" >> "$OUT_PATH"/log
    #echo "---new--- re-rerun w/ simple-mlp" >> "$OUT_PATH"/log
    #echo "+++new+++ re-rerun w/ tiny unet" >> "$OUT_PATH"/log
    #echo "new/test re-rerun w/ tiny unet" >> "$OUT_PATH"/log
    echo "new/test re-rerun w/ simple-mlp" >> "$OUT_PATH"/log
done