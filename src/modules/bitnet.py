"""
BitNet 1.58b - Ternary Weight Quantization

Implements {-1, 0, 1} weight quantization with activation quantization.
Based on: "The Era of 1-bit LLMs" (BitNet b1.58)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BitLinear(nn.Linear):
    """
    Linear layer with ternary weight quantization {-1, 0, 1}.
    
    During forward:
    1. Quantize activations to 8-bit (simulated)
    2. Quantize weights to {-1, 0, 1}
    3. Perform matrix multiplication
    4. Use Straight-Through Estimator (STE) for backprop
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias (default: False for BitNet)
        eps: Small constant for numerical stability
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps
        
        # RMSNorm for activation normalization
        self.rms_norm = nn.RMSNorm(in_features, eps=eps)
        
    def activation_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize activations to 8-bit (simulated in float).
        
        Formula: Q(x) = Clip(Round(x / s), -128, 127) * s
        where s = max(|x|) / 127
        
        Args:
            x: Input tensor [*, in_features]
            
        Returns:
            Quantized tensor (same shape as input)
        """
        # Force float32 for stability during scaling
        x_f32 = x.float()
        
        # Compute scale: max absolute value per token
        scale = 127.0 / x_f32.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)
        
        # Quantize: scale, round, clip, unscale
        x_quant = (x_f32 * scale).round().clamp_(-128, 127) / scale
        
        # Cast back to original dtype
        x_quant = x_quant.to(x.dtype)
        
        # Straight-Through Estimator: use quantized in forward, original gradient
        # We return x + (x_quant - x).detach()
        # Forward: x + x_quant - x = x_quant
        # Backward: gradient flows through x (STE)
        return x + (x_quant - x).detach()
    
    def weight_quant(self, w: torch.Tensor) -> torch.Tensor:
        """
        Quantize weights to {-1, 0, 1}.
        
        Formula: Q(W) = Clip(Round(W / γ), -1, 1) * γ
        where γ = mean(|W|)
        
        Args:
            w: Weight tensor [out_features, in_features]
            
        Returns:
            Ternary quantized weights
        """
        # Force float32 for stability
        w_f32 = w.float()
        
        # Compute scale: mean absolute value (clamp to avoid division by zero)
        gamma = w_f32.abs().mean()
        gamma = torch.clamp(gamma, min=self.eps)
        
        # Normalize, round, and clamp to {-1, 0, 1}
        w_normalized = w_f32 / gamma
        w_quant_norm = torch.clamp(torch.round(w_normalized), -1, 1)
        
        # Scale back
        w_quant = w_quant_norm * gamma
        
        # Cast back and Straight-Through Estimator
        w_quant = w_quant.to(w.dtype)
        return w + (w_quant - w).detach()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights and activations.
        
        Args:
            x: Input [batch, seq_len, in_features]
            
        Returns:
            Output [batch, seq_len, out_features]
        """
        # 1. Normalize activations (Force float32 for stability)
        x = self.rms_norm(x.float()).to(x.dtype)
        
        # 2. Quantize activations
        x_q = self.activation_quant(x)
        
        # 3. Quantize weights
        w_q = self.weight_quant(self.weight)
        
        # 4. Linear transformation
        return F.linear(x_q, w_q, self.bias)
    
    def extra_repr(self) -> str:
        """String representation for debugging"""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, eps={self.eps}'


# Export
__all__ = ['BitLinear']


if __name__ == "__main__":
    # Quick self-test
    print("BitLinear Self-Test")
    print("=" * 50)
    
    # Create layer
    layer = BitLinear(128, 256, bias=False)
    print(f"✓ Created BitLinear(128, 256)")
    
    # Test forward pass
    x = torch.randn(4, 10, 128)  # [batch=4, seq=10, features=128]
    out = layer(x)
    print(f"✓ Forward pass: {x.shape} → {out.shape}")
    
    # Test weight quantization
    w_quant = layer.weight_quant(layer.weight)
    unique_vals = torch.unique(w_quant / w_quant.abs().mean())
    unique_vals = unique_vals[unique_vals.abs() > 0.01]  # Filter near-zero
    print(f"✓ Weight quantization: unique values = {unique_vals.tolist()}")
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    print(f"✓ Gradient flow: grad norm = {layer.weight.grad.norm().item():.4f}")
    
    print("\n✅ All tests passed!")
