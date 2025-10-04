#!/bin/bash

# Fixed training script with improved hyperparameters to prevent model collapse
echo "Starting improved SeqXGPT training..."
echo "Fixes applied:"
echo "  - Consistent CRF training/inference"
echo "  - Class weight balancing" 
echo "  - Gradient clipping"
echo "  - Better learning rate and regularization"
echo "  - Layer normalization"

# First run data analysis
echo "Analyzing training data..."
python analyze_data.py

echo -e "\nStarting training with improved parameters..."
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