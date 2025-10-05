#!/bin/bash

# Working training script - CRF issues fixed
echo "Starting SeqXGPT training with fixed model..."
echo "Changes:"
echo "  - Disabled CRF (was causing API errors)"
echo "  - Using weighted CrossEntropy for class balancing"
echo "  - Stable training with gradient clipping"

python src/model/train.py \
    --model=Simple \
    --gpu=0 \
    --batch_size=8 \
    --seq_len=256 \
    --num_train_epochs=20 \
    --lr=2e-5 \
    --warm_up_ratio=0.1 \
    --weight_decay=0.01 \
    --train_path=train_data.jsonl \
    --test_path=test_data.jsonl

echo "Training completed!"