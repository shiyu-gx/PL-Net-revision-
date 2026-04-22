#!/bin/bash
set -e

# Ablation study for PL-Net

epochs=30
target_size=1024

# baseline
python train_linux.py \
    --epochs ${epochs} \
    --target_size ${target_size} \
    --exp_name "baseline" \
    --in_channels 3 \
    --use_sam False \
    --fusion_method "none" \
    --loss_function "cross_entropy"

# baseline + DEM
python train_linux.py \
    --epochs ${epochs} \
    --target_size ${target_size} \
    --exp_name "baseline_dem" \
    --in_channels 4 \
    --use_sam False \
    --fusion_method "none" \
    --loss_function "cross_entropy"

# baseline + DEM + SAM
python train_linux.py \
    --epochs ${epochs} \
    --target_size ${target_size} \
    --exp_name "baseline_dem_sam" \
    --in_channels 4 \
    --use_sam True \
    --fusion_method "add" \
    --loss_function "cross_entropy"

# baseline + DEM + SAM + CSAF
python train_linux.py \
    --epochs ${epochs} \
    --target_size ${target_size} \
    --exp_name "baseline_dem_sam_csaf" \
    --in_channels 4 \
    --use_sam True \
    --fusion_method "csaf" \
    --loss_function "cross_entropy"

# baseline + DEM + SAM + CSAF + TCS
python train_linux.py \
    --epochs ${epochs} \
    --target_size ${target_size} \
    --exp_name "baseline_dem_sam_csaf_tcs" \
    --in_channels 4 \
    --use_sam True \
    --fusion_method "csaf" \
    --loss_function "tcs"
