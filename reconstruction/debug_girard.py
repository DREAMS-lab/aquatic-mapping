#!/usr/bin/env python3
"""Debug script to understand Girard kernel computation"""
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loading import load_sample_data
from girard_uncertain_input import ExpectedRBFKernel

# Load training data
data_dir = Path(__file__).parent.parent / 'src' / 'sampling' / 'data' / 'missions'
csv_path = data_dir / 'trial_1' / 'radial' / 'radial_samples.csv'

x_samples, y_samples, temp_samples, position_cov = load_sample_data(csv_path)
print(f"Loaded {len(x_samples)} samples")
print(f"Position covariance shape: {position_cov.shape}")
print(f"Cov_xx range: [{position_cov[:, 0].min():.6f}, {position_cov[:, 0].max():.6f}]")
print(f"Cov_yy range: [{position_cov[:, 2].min():.6f}, {position_cov[:, 2].max():.6f}]")

# Create kernel
train_x = torch.tensor(np.c_[x_samples, y_samples], dtype=torch.float32)
train_cov = torch.tensor(position_cov, dtype=torch.float32)

kernel = ExpectedRBFKernel(train_cov=train_cov)
kernel.lengthscale = torch.tensor(1.0)

# Test diagonal
print("\n--- Testing Diagonal ---")
diag = kernel._expected_diag(train_x, kernel.lengthscale)
print(f"Diagonal shape: {diag.shape}")
print(f"Diagonal min/max: {diag.min():.6f} / {diag.max():.6f}")
print(f"Diagonal mean: {diag.mean():.6f}")
print(f"Sample diagonal values: {diag[:5]}")

# Test kernel computation on small subset
print("\n--- Testing Kernel on Subset ---")
subset_size = 5
x_subset = train_x[:subset_size]
K_subset = kernel._expected_kernel(x_subset, x_subset, kernel.lengthscale)
print(f"Kernel subset shape: {K_subset.shape}")
print(f"Kernel subset:\n{K_subset}")
print(f"Diagonal of K_subset: {torch.diag(K_subset)}")

# Test full kernel
print("\n--- Testing Full Kernel (Training-Training) ---")
K_full = kernel._expected_kernel(train_x, train_x, kernel.lengthscale)
print(f"Full kernel shape: {K_full.shape}")
print(f"Full kernel min/max: {K_full.min():.6f} / {K_full.max():.6f}")
print(f"Full kernel diagonal min/max: {torch.diag(K_full).min():.6f} / {torch.diag(K_full).max():.6f}")

# Check if kernel is positive definite
try:
    L = torch.linalg.cholesky(K_full)
    print("Kernel is positive definite ✓")
except Exception as e:
    print(f"Kernel is NOT positive definite: {e}")

# Test training-test kernel
print("\n--- Testing Kernel (Training-Test) ---")
test_x = torch.randn(10, 2)
K_train_test = kernel._expected_kernel(train_x, test_x, kernel.lengthscale)
print(f"Training-test kernel shape: {K_train_test.shape}")
print(f"Training-test kernel min/max: {K_train_test.min():.6f} / {K_train_test.max():.6f}")

print("\nDone!")
