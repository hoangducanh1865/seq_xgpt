#!/bin/bash

# Quick fix training script to handle data collator issues
echo "Starting SeqXGPT training with data fixes..."

python src/model/train.py \
    --model=Simple \
    --gpu=1 \
    --batch_size=4 \
    --seq_len=128 \
    --num_train_epochs=10 \
    --lr=1e-5 \
    --warm_up_ratio=0.1 \
    --weight_decay=0.01 \
    --train_path=train_data.jsonl \
    --test_path=test_data.jsonl

echo "Training completed!"