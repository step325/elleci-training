"""
Elleci V1 - Quick Inference Test
Test a checkpoint to see if the model can generate coherent text.
"""
import torch
import argparse
from transformers import PreTrainedTokenizerFast
from src.config import NanoPrimeConfig
from src.model import NanoPrime

def load_model(checkpoint_path, device="cuda"):
    """Load model from checkpoint."""
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    # Initialize config and model
    config = NanoPrimeConfig()
    model = NanoPrime(config)
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    return model

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """Generate text from prompt."""
    device = next(model.parameters()).device
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            # Handle model output - could be tuple or tensor
            if isinstance(outputs, tuple):
                logits = outputs[0][:, -1, :] / temperature
            else:
                logits = outputs[:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/elleci_step_10000.pth")
    parser.add_argument("--prompt", type=str, default="Ciao, come stai?")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_chimera_v2_patched")
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Test prompts
    test_prompts = [
        args.prompt,
        "Spiegami cos'è la fotosintesi.",
        "Scrivi una breve poesia sul mare.",
        "What is machine learning?",
        "<|im_start|>user\nCome ti chiami?<|im_end|>\n<|im_start|>assistant\n",
    ]
    
    print("\n" + "="*60)
    print("🧪 TESTING MODEL GENERATION")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\n📌 Prompt: {prompt[:50]}...")
        print("-"*40)
        
        output = generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
        print(f"🤖 Output:\n{output}")
        print("-"*40)

if __name__ == "__main__":
    main()
