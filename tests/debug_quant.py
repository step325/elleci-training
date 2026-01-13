"""Debug script for weight quantization"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.modules.bitnet import BitLinear

# Create layer
layer = BitLinear(8, 16)  # Smaller for debugging
w = layer.weight.clone()

print("Original weights (first 5):", w[0, :5])

# Quantize
w_quant = layer.weight_quant(w)
print("Quantized weights (first 5):", w_quant[0, :5])

# Check normalization
gamma = w.abs().mean()
print(f"Gamma: {gamma.item()}")

w_norm = w_quant / gamma
print("Normalized quantized (first 10):", w_norm.flatten()[:10])

unique = torch.unique(w_norm.round())
print(f"Unique values after rounding: {unique.tolist()}")

# Check if truly ternary
w_norm_check = w / gamma
w_quant_check = w_norm_check.round().clamp(-1, 1) * gamma
print("Manual quant (first 5):", w_quant_check[0, :5])

print(f"\nAre they equal? {torch.allclose(w_quant, w_quant_check)}")
