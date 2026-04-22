#!/bin/bash
set -e

# DEM threshold sensitivity analysis
# Usage:
#   bash scripts/dem_threshold_sensitivity.sh 5.85 5.90 5.95

epochs=30
target_size=256

if [ "$#" -eq 0 ]; then
    echo "Please provide one or more threshold values."
    echo "Example: bash scripts/dem_threshold_sensitivity.sh 5.85 5.90 5.95"
    exit 1
fi

for threshold in "$@"
do
    python train_linux.py \
        --epochs ${epochs} \
        --target_size ${target_size} \
        --exp_name "dem_threshold_${threshold}" \
        --in_channels 4 \
        --use_sam True \
        --fusion_method "csaf" \
        --loss_function "tcs" \
        --threshold_m ${threshold}
done
