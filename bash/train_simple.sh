#!/bin/bash

# Improved training script for text-only data with better hyperparameters
python src/model/train.py \
    --model=Simple \
    --gpu=0 \
    --batch_size=8 \
    --seq_len=256 \
    --num_train_epochs=50 \
    --lr=5e-5 \
    --warm_up_ratio=0.1 \
    --weight_decay=0.01 \
    --train_path=train_data.jsonl \
    --test_path=test_data.jsonl