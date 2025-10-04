#!/bin/bash

# Simple testing script for text-only data
python src/model/train.py \
    --model=Simple \
    --gpu=0 \
    --do_test \
    --batch_size=16 \
    --seq_len=512 \
    --train_path=train_data.jsonl \
    --test_path=test_data.jsonl