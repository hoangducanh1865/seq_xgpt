# SeqXGPT - Text-Only Mode

This guide shows how to use SeqXGPT with raw text data, skipping the complex feature extraction phase.

## Data Format

Your training and test data should be in JSONL format with the following fields:

```json
{
  "text": "Your text content here...",
  "label": "gpt2|gptneo|gptj|llama|gpt3re|human",
  "prompt_len": 123  // optional - will be estimated if missing
}
```

## Training

Use the Simple model for text-only training:

```bash
# Train with simple text classifier (uses GPU by default)
bash bash/train_simple.sh

# Or run directly:
python src/model/train.py \
    --model=Simple \
    --train_path=train_data.jsonl \
    --test_path=test_data.jsonl \
    --batch_size=16 \
    --seq_len=512 \
    --num_train_epochs=3 \
    --gpu=0
```

## Testing

```bash
# Test the trained model
bash bash/test_simple.sh

# Or run directly:
python src/model/train.py \
    --model=Simple \
    --do_test \
    --train_path=train_data.jsonl \
    --test_path=test_data.jsonl
```

## Features

The Simple model automatically extracts basic features from raw text:
- Token length (normalized)
- Relative position in sequence
- Uppercase indicator
- Number presence indicator

No need for complex perplexity calculation or multiple model APIs!

## Model Options

- `--model=Simple`: Text-based classifier (recommended for raw text)
- `--model=CNN`: Original CNN model (requires feature extraction)
- `--model=Transformer`: Original Transformer model (requires feature extraction)
- `--model=RNN`: RNN model (requires feature extraction)

## Troubleshooting

### CUDA Issues Diagnosis

If you encounter CUDA errors, first run the diagnostic script:

```bash
python debug_cuda.py
```

This will check your CUDA environment and test GPU operations.

### Common CUDA Fixes

1. **"No kernel image available"**: Usually a PyTorch/CUDA version mismatch
   ```bash
   # Reinstall PyTorch with correct CUDA version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Out of memory**: Reduce batch size and sequence length
   ```bash
   # Use smaller parameters
   --batch_size=8 --seq_len=256
   ```

3. **Driver issues**: Update NVIDIA drivers or check compatibility

### CPU Fallback (if needed)

Only use CPU mode as a last resort:
```bash
bash bash/train_simple_cpu.sh
```

## Performance

The Simple model trades some accuracy for significant simplicity:
- ✅ No feature extraction servers needed
- ✅ Works with any text dataset
- ✅ Much faster training and inference
- ✅ CPU fallback for compatibility
- ⚠️ May have lower accuracy than full SeqXGPT pipeline