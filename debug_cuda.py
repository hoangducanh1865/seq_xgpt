#!/usr/bin/env python3
"""
CUDA Debugging Script for SeqXGPT
This script helps diagnose CUDA compatibility issues.
"""

import torch
import os
import subprocess
import sys

def check_cuda_environment():
    """Check CUDA environment and compatibility"""
    print("=" * 60)
    print("CUDA Environment Check")
    print("=" * 60)
    
    # Basic CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - Total Memory: {props.total_memory / 1024**3:.1f} GB")
    
    # System CUDA info
    print("\n" + "=" * 60)
    print("System CUDA Check")
    print("=" * 60)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout)
        else:
            print("nvidia-smi not available or failed")
    except FileNotFoundError:
        print("nvidia-smi command not found")
    
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("NVCC version:")
            print(result.stdout)
        else:
            print("NVCC not available")
    except FileNotFoundError:
        print("NVCC command not found")

def test_cuda_operations():
    """Test basic CUDA operations"""
    print("\n" + "=" * 60)
    print("CUDA Operations Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return
    
    try:
        # Test 1: Simple tensor creation
        print("Test 1: Creating tensors on GPU...")
        device = torch.device('cuda')
        x = torch.randn(10, 10).to(device)
        print("✓ Tensor creation successful")
        
        # Test 2: Basic operations
        print("Test 2: Basic operations...")
        y = torch.randn(10, 10).to(device)
        z = x + y
        print("✓ Addition successful")
        
        # Test 3: ReLU (the operation that was failing)
        print("Test 3: ReLU operation...")
        a = torch.randn(10, 10).to(device)
        b = torch.relu(a)
        print("✓ ReLU successful")
        
        # Test 4: Linear layer
        print("Test 4: Linear layer...")
        linear = torch.nn.Linear(10, 5).to(device)
        output = linear(x)
        print("✓ Linear layer successful")
        
        # Test 5: Backward pass
        print("Test 5: Backward pass...")
        loss = output.sum()
        loss.backward()
        print("✓ Backward pass successful")
        
        print("\n✅ All CUDA tests passed!")
        
    except Exception as e:
        print(f"\n❌ CUDA test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False
    
    return True

def suggest_solutions():
    """Suggest solutions based on the environment"""
    print("\n" + "=" * 60)
    print("Suggestions")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Possible solutions:")
        print("1. Install CUDA-enabled PyTorch")
        print("2. Check if NVIDIA drivers are installed")
        print("3. Use CPU mode with --gpu=-1")
    else:
        print("CUDA is available but operations might fail. Try:")
        print("1. Update NVIDIA drivers")
        print("2. Reinstall PyTorch with matching CUDA version")
        print("3. Check for compute capability compatibility")
        print("4. Try different PyTorch/CUDA version combinations")
        
        print("\nRecommended PyTorch installation commands:")
        print("# For CUDA 11.8:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n# For CUDA 12.1:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    check_cuda_environment()
    success = test_cuda_operations()
    suggest_solutions()
    
    if success:
        print("\n🎉 CUDA is working correctly!")
        print("You can proceed with GPU training using:")
        print("bash bash/train_simple.sh")
    else:
        print("\n⚠️  CUDA issues detected.")
        print("Consider using CPU mode or fixing CUDA installation.")
        print("For CPU training: bash bash/train_simple_cpu.sh")