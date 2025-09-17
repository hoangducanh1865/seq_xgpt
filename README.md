# SeqXGPT Baseline

# Run servers
```
First you need to run all the servers with server scripts in folder ./server_scripts/
```

# Split data
```
python src/model/train.py \
    --gpu=0 \
    --split_dataset \
    --data_path=src/dataset/SeqXGPT-Bench/ \
    --train_path=train_data.jsonl \
    --test_path=test_data.jsonl
```

```
python src/model/train.py \
    --gpu=0 \
    --data_path=src/dataset/SeqXGPT-Bench/ \
    --train_path=train_data_with_features.jsonl \
    --test_path=test_data_with_features.jsonl
```

# Extract features
```
python src/dataset/gen_features.py --get_en_features \
    --input_file train_data.jsonl \
    --output_file train_data_with_features.jsonl
```

```
python src/dataset/gen_features.py --get_en_features \
    --input_file test_data.jsonl \
    --output_file test_data_with_features.jsonl
```

# Train 
```
python src/model/train.py \
    --gpu=0 \
    --train_path=train_data_with_features.jsonl \
    --test_path=test_data_with_features.jsonl
```

# Evaluate in sentence-level
```
python src/model/train.py \
    --gpu=0 \
    --do_test \
    --train_path=train_data_with_features.jsonl \
    --test_path=test_data_with_features.jsonl
```

# Evaluate in content-level
```
python src/model/train.py \
    --gpu=0 \
    --do_test \
    --test_content \
    --train_path=train_data_with_features.jsonl \
    --test_path=test_data_with_features.jsonl
```