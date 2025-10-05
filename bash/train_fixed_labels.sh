#!/bin/bash

# Fixed training script with corrected label mapping
echo "Starting SeqXGPT training with fixed label mappings..."
echo "Fixed issues:"
echo "  - Proper 6-class label mapping (gpt2, gptneo, gptj, llama, gpt3re, human)"
echo "  - Simplified sequence labeling (no BMES complexity)"
echo "  - Direct label assignment per sequence"

python src/model/train.py \
    --model=Simple \
    --gpu=0 \
    --batch_size=4 \
    --seq_len=128 \
    --num_train_epochs=5 \
    --lr=1e-5 \
    --warm_up_ratio=0.1 \
    --weight_decay=0.01 \
    --train_path=train_data.jsonl \
    --test_path=test_data.jsonl

echo "Training completed!"