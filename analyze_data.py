#!/usr/bin/env python3
"""
Data Analysis Script for SeqXGPT Training
Analyzes class distribution and data patterns to help debug training issues.
"""

import json
from collections import Counter
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def analyze_data(file_path):
    """Analyze JSONL data file"""
    print(f"\n{'='*60}")
    print(f"Analyzing {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist!")
        return
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Total samples: {len(data)}")
    
    # Analyze labels
    text_labels = [item['label'] for item in data]
    label_counts = Counter(text_labels)
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = count / len(data) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Check for class imbalance
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 10:
        print("⚠️  SEVERE class imbalance detected!")
    elif imbalance_ratio > 3:
        print("⚠️  Moderate class imbalance detected")
    
    # Analyze text lengths
    text_lengths = [len(item['text'].split()) for item in data]
    print(f"\nText length statistics:")
    print(f"  Min: {min(text_lengths)} words")
    print(f"  Max: {max(text_lengths)} words") 
    print(f"  Average: {sum(text_lengths)/len(text_lengths):.1f} words")
    
    # Show sample data
    print(f"\nFirst 3 samples:")
    for i, item in enumerate(data[:3]):
        print(f"  Sample {i+1}:")
        print(f"    Label: {item['label']}")
        print(f"    Text length: {len(item['text'].split())} words")
        print(f"    Text preview: {item['text'][:100]}...")
        if 'prompt_len' in item:
            print(f"    Prompt length: {item['prompt_len']}")
    
    return data, label_counts

def check_label_mapping():
    """Check label mapping consistency"""
    print(f"\n{'='*60}")
    print("Checking label mappings")
    print(f"{'='*60}")
    
    # Import the label mappings
    try:
        from utils.backend_model_info import en_labels, id2label
        print("Label mappings loaded successfully:")
        print(f"en_labels: {en_labels}")
        print(f"id2label: {id2label}")
        
        # Check if labels are consistent
        expected_labels = set(en_labels)
        return expected_labels, id2label
    except ImportError as e:
        print(f"Could not import label mappings: {e}")
        return None, None

def suggest_fixes(train_data, test_data, label_counts):
    """Suggest fixes for training issues"""
    print(f"\n{'='*60}")
    print("Training Issue Analysis & Suggestions")
    print(f"{'='*60}")
    
    issues = []
    suggestions = []
    
    # Check class imbalance
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 5:
        issues.append("Severe class imbalance")
        suggestions.append("Use class weights in loss function")
        suggestions.append("Apply data augmentation for minority classes")
        suggestions.append("Use focal loss instead of cross-entropy")
    
    # Check dataset size
    if len(train_data) < 1000:
        issues.append("Small training dataset")
        suggestions.append("Reduce model complexity")
        suggestions.append("Use stronger regularization")
        suggestions.append("Consider data augmentation")
    
    # Check for single class dominance
    dominant_class_pct = max(label_counts.values()) / len(train_data) * 100
    if dominant_class_pct > 80:
        issues.append(f"One class dominates {dominant_class_pct:.1f}% of data")
        suggestions.append("This explains why model predicts only one class!")
        suggestions.append("Balance the dataset or use class weights")
    
    print("Issues detected:")
    for issue in issues:
        print(f"  ❌ {issue}")
    
    print("\nSuggested fixes:")
    for suggestion in suggestions:
        print(f"  ✅ {suggestion}")
    
    # Calculate ideal class weights
    if len(label_counts) > 1:
        print(f"\nRecommended class weights:")
        total_samples = sum(label_counts.values())
        num_classes = len(label_counts)
        for label, count in sorted(label_counts.items()):
            weight = total_samples / (num_classes * count)
            print(f"  {label}: {weight:.2f}")

if __name__ == "__main__":
    # Analyze training data
    train_data, train_labels = analyze_data("train_data.jsonl")
    
    # Analyze test data  
    test_data, test_labels = analyze_data("test_data.jsonl")
    
    # Check label mappings
    expected_labels, id2label = check_label_mapping()
    
    # Suggest fixes
    if train_data and train_labels:
        suggest_fixes(train_data, test_data, train_labels)
    
    print(f"\n{'='*60}")
    print("Next Steps:")
    print("1. Fix class imbalance if detected")
    print("2. Use smaller learning rate (5e-5 or 1e-5)")
    print("3. Add gradient clipping")
    print("4. Use proper class weights")
    print("5. Monitor training loss carefully")
    print(f"{'='*60}")