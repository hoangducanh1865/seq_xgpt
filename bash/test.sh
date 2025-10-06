#!/bin/bash
# Testing script - loads pretrained model and evaluates on test set

echo "🔍 Starting testing workflow..."
echo "===================================="

# Check if pretrained model exists
MODEL_PATH="data/output_models/transformer_faid_three_labels_final.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Pretrained model not found at $MODEL_PATH"
    echo "Please run training first using: bash bash/train.sh"
    exit 1
fi

python src/model/train.py \
    --setup 'three_labels' \
    --dataset 'faid' \
    --gpu=1 \
    --do_test \
    --load_pretrained \
    --test_content

echo "✅ Testing workflow completed!" 