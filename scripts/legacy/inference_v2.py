"""
Elleci v2 - Production Inference Script

Clean, fast inference without router complexity.
Optimized for the "Fine Ultimo" vision: efficiency + reasoning.
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import ElleciConfig
from src.model import Elleci
from transformers import AutoTokenizer


def load_v2_model(checkpoint_path='nanoprime_v2_final.pth'):
    """Load production v2 model"""
    print("Loading Elleci v2 (Production)...")
    
    config = ElleciConfig()
    config.n_layers = 6
    config.max_seq_len = 64
    
    model = Elleci(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"✓ Device: {config.device}")
    print(f"✓ Architecture: BitNet + Mamba + MLA (Hybrid)")
    
    return model, config


def generate(model, tokenizer, config, prompt="", max_tokens=50, temperature=0.8, top_k=50):
    """Generate text - production optimized"""
    
    # Encode prompt
    if prompt:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=config.device)
    else:
        input_ids = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    
    # Generate (no router - pure v2)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    return tokenizer.decode(output[0].tolist(), skip_special_tokens=True)


def interactive():
    """Simple interactive loop"""
    print("\n" + "=" * 70)
    print("Elleci v2 - Production Inference")
    print("=" * 70)
    print("\nCommands:")
    print("  [Enter] - Generate story")
    print("  'temp: X' - Set temperature (0.1-2.0)")
    print("  'quit' - Exit")
    print("=" * 70)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model, config = load_v2_model()
    
    temp = 0.8
    
    while True:
        cmd = input("\n> ").strip()
        
        if cmd.lower() in ['quit', 'exit', 'q']:
            break
        
        if cmd.lower().startswith('temp:'):
            try:
                temp = float(cmd.split(':')[1].strip())
                temp = max(0.1, min(2.0, temp))
                print(f"✓ Temperature = {temp}")
                continue
            except:
                print("❌ Invalid. Use: temp: 0.8")
                continue
        
        # Generate
        prompt = cmd if cmd else ""
        story = generate(model, tokenizer, config, prompt, max_tokens=50, temperature=temp)
        print(f"\n📖 {story}")


def interactive(checkpoint_path='nanoprime_v2_final.pth'):
    """Interactive generation loop"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model, config = load_v2_model(checkpoint_path)
    
    print("-" * 50)
    print("Elleci v2 Interactive Mode")
    print(f"Loaded: {checkpoint_path}")
    print("Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            cmd = input("\nPropmt: ")
        except EOFError:
            break
            
        if cmd.lower() in ['quit', 'exit', 'q']:
            break
        
        # ... (rest is same, just need to ensure correct flow)

def batch_test(n=5, checkpoint_path='nanoprime_v2_final.pth'):
    """Quick quality test"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model, config = load_v2_model(checkpoint_path)
    
    print(f"\nGenerating {n} samples from {checkpoint_path}...\n")
    
    for i in range(n):
        story = generate(model, tokenizer, config, max_tokens=40, temperature=0.9)
        print(f"{i+1}. {story}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=0, help='Generate N samples')
    parser.add_argument('--checkpoint', type=str, default='nanoprime_v2_final.pth')
    args = parser.parse_args()
    
    if args.batch > 0:
        batch_test(args.batch, args.checkpoint)
    else:
        interactive(args.checkpoint)
