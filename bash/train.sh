#!/bin/bash
# Training script - trains model and shows results on test set

echo "🚀 Starting training workflow..."
echo "===================================="

python src/model/train.py \
    --setup 'three_labels' \
    --dataset 'faid' \
    --gpu=0 \
    --num_train_epochs 10 \
    --batch_size 8 \
    --lr 5e-5

echo "✅ Training workflow completed!"