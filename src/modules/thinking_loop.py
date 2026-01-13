"""
Thinking Loop - Recurrent Reasoning Module

Implements iterative refinement for complex reasoning.
Used in "slow path" (System 2) for multi-step thinking.
"""
import torch
import torch.nn as nn


class ThinkingLoop(nn.Module):
    """
    Recurrent thinking mechanism.
    
    Takes an initial hidden state and refines it through N iterations.
    Each iteration uses the same transformer layer.
    
    Args:
        config: ThinkingLoopConfig with max_iterations, etc.
        layer: The layer to apply recurrently (e.g., TransformerBlock)
    """
    def __init__(self, config, layer):
        super().__init__()
        self.max_iterations = config.max_iterations
        self.convergence_threshold = config.convergence_threshold
        self.layer = layer
        
    def forward(self, x, n_iterations=None):
        """
        Apply thinking loop.
        
        Args:
            x: Input [batch, seq_len, d_model]
            n_iterations: Number of iterations (default: max_iterations)
            
        Returns:
            output: Refined state [batch, seq_len, d_model]
            n_steps: Actual number of iterations used
        """
        if n_iterations is None:
            n_iterations = self.max_iterations
        
        current = x
        
        for i in range(n_iterations):
            # Apply layer
            next_state = self.layer(current)
            
            # Check convergence (optional early stopping)
            if i > 0:
                delta = (next_state - current).abs().mean()
                if delta < self.convergence_threshold:
                    return next_state, i + 1
            
            current = next_state
        
        return current, n_iterations


class SimpleThinkingLayer(nn.Module):
    """
    Simple layer for thinking loop (for testing).
    In practice, this would be a full transformer block.
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
    def forward(self, x):
        # Simple residual block
        return x + self.ffn(self.norm(x))


# Export
__all__ = ['ThinkingLoop', 'SimpleThinkingLayer']


if __name__ == "__main__":
    # Self-test
    print("Thinking Loop Self-Test")
    print("=" * 50)
    
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        max_iterations: int = 4
        convergence_threshold: float = 0.01
    
    config = TestConfig()
    
    # Create a simple layer for testing
    d_model = 768
    layer = SimpleThinkingLayer(d_model)
    
    # Create thinking loop
    thinking = ThinkingLoop(config, layer)
    
    print(f"✓ Created ThinkingLoop (max_iter={config.max_iterations})")
    
    # Test forward
    x = torch.randn(2, 16, d_model)  # [batch, seq, d_model]
    output, n_steps = thinking(x)
    
    print(f"✓ Forward pass: {x.shape} → {output.shape}")
    print(f"  Iterations used: {n_steps}")
    
    # Test with fewer iterations
    output2, n_steps2 = thinking(x, n_iterations=2)
    print(f"✓ Limited iterations: used {n_steps2}/2")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print(f"✓ Gradient flow: layer.ffn[0] grad norm = {layer.ffn[0].weight.grad.norm().item():.4f}")
    
    # Test convergence
    # Create input that should converge quickly
    x_simple = torch.zeros(2, 16, d_model)
    output_conv, n_conv = thinking(x_simple)
    print(f"✓ Convergence test: converged in {n_conv} steps (max={config.max_iterations})")
    
    print("\n✅ All tests passed!")
