#!/bin/bash

# CPU-only training script for environments with CUDA compatibility issues
# This version is specifically for Kaggle or other cloud environments

echo "Starting CPU-only training..."
echo "This may be slower but avoids CUDA compatibility issues"

# Set environment to force CPU mode
export CUDA_VISIBLE_DEVICES=""

python src/model/train.py \
    --model=Simple \
    --gpu=-1 \
    --batch_size=8 \
    --seq_len=256 \
    --num_train_epochs=100 \
    --lr=1e-4 \
    --train_path=train_data.jsonl \
    --test_path=test_data.jsonl

echo "Training completed!"