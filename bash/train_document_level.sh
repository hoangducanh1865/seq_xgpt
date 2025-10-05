#!/bin/bash

# Document-level classification training script
echo "Starting SeqXGPT Document-Level Classification Training..."
echo "Improvements:"
echo "  - Document-level classification (not token-level)"
echo "  - Proper pooling and aggregation"
echo "  - Better evaluation metrics"
echo "  - Optimized for classification accuracy"

python src/model/train.py \
    --model=Simple \
    --gpu=0 \
    --batch_size=16 \
    --seq_len=128 \
    --num_train_epochs=10 \
    --lr=1e-4 \
    --warm_up_ratio=0.1 \
    --weight_decay=0.01 \
    --train_path=train_data.jsonl \
    --test_path=test_data.jsonl

echo "Training completed!"