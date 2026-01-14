"""
Elleci V1 - Comprehensive Overfitting Diagnosis
Tests multiple hypotheses for why the model generates repetitive tokens.
"""
import torch
import numpy as np
from collections import Counter
from transformers import PreTrainedTokenizerFast
from src.config import ElleciConfig
from src.model import Elleci
import argparse

def load_model(checkpoint_path, device="cuda"):
    """Load model from checkpoint."""
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    config = ElleciConfig()
    model = Elleci(config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    return model, config

def generate_with_params(model, tokenizer, prompt, max_tokens=50, temperature=1.0, 
                         top_k=0, top_p=1.0, repetition_penalty=1.0):
    """Generate with configurable parameters."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            next_logits = logits[0, -1, :].clone()
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist() + generated_ids):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated_ids)

def test_generation_params(model, tokenizer):
    """Test various generation parameter combinations."""
    print("\n" + "="*70)
    print("🧪 TEST 1: Generation Parameter Sweep")
    print("="*70)
    
    prompts = ["Ciao, come stai?", "La capitale d'Italia è"]
    
    param_combos = [
        {"temperature": 1.0, "top_k": 0, "repetition_penalty": 1.0, "name": "Greedy"},
        {"temperature": 0.7, "top_k": 0, "repetition_penalty": 1.0, "name": "Temp=0.7"},
        {"temperature": 0.7, "top_k": 40, "repetition_penalty": 1.0, "name": "Temp=0.7 + TopK=40"},
        {"temperature": 0.7, "top_k": 40, "repetition_penalty": 1.2, "name": "Temp=0.7 + TopK=40 + RepPen=1.2"},
        {"temperature": 0.7, "top_k": 40, "repetition_penalty": 1.5, "name": "Temp=0.7 + TopK=40 + RepPen=1.5"},
        {"temperature": 1.0, "top_k": 0, "repetition_penalty": 2.0, "name": "RepPen=2.0 only"},
    ]
    
    for prompt in prompts:
        print(f"\n📌 Prompt: '{prompt}'")
        print("-"*60)
        for params in param_combos:
            output = generate_with_params(
                model, tokenizer, prompt, max_tokens=30,
                temperature=params["temperature"],
                top_k=params["top_k"],
                repetition_penalty=params["repetition_penalty"]
            )
            # Check if repetitive
            unique_chars = len(set(output))
            is_repetitive = unique_chars < 5 or len(output) < 3
            status = "❌ REP" if is_repetitive else "✅ OK"
            print(f"  {params['name']:<35} {status} → '{output[:40]}...'")

def analyze_lm_head_weights(model, tokenizer):
    """Analyze the language model head for bias."""
    print("\n" + "="*70)
    print("🧪 TEST 2: LM Head Weight Analysis")
    print("="*70)
    
    lm_head = model.lm_head
    weights = lm_head.weight.data.float()
    
    # Compute per-token "bias" (norm of weight vector)
    token_norms = weights.norm(dim=1)
    
    # Top tokens by weight norm
    top_indices = torch.topk(token_norms, 20).indices
    print("\nTop 20 tokens by weight norm (potential bias):")
    for i, idx in enumerate(top_indices):
        token = tokenizer.decode([idx.item()])
        norm = token_norms[idx].item()
        print(f"  {i+1:2d}. '{token}' (norm={norm:.4f})")
    
    # Check common tokens
    print("\nCommon token weight norms:")
    for token in [",", ".", " ", "è", "the", "a", "\n"]:
        token_id = tokenizer.encode(token, add_special_tokens=False)
        if token_id:
            norm = token_norms[token_id[0]].item()
            print(f"  '{token}' (id={token_id[0]}): norm={norm:.4f}")

def inspect_dataset_samples(tokenizer, n_samples=10):
    """Inspect actual dataset samples."""
    print("\n" + "="*70)
    print("🧪 TEST 3: Dataset Sample Inspection")
    print("="*70)
    
    try:
        from data.elleci_dataset import EllediDataset
        dataset = EllediDataset(tokenizer, seq_len=256, buffer_size=100)
        
        # Collect samples
        print(f"\nFirst {n_samples} training samples (decoded):")
        print("-"*60)
        
        for i in range(n_samples):
            sample = next(iter(dataset))
            text = tokenizer.decode(sample[:100], skip_special_tokens=True)
            # Check for weird patterns
            comma_ratio = text.count(',') / max(len(text), 1)
            print(f"Sample {i+1}: '{text[:80]}...' (comma ratio: {comma_ratio:.2%})")
            
    except Exception as e:
        print(f"⚠️ Could not load dataset: {e}")

def analyze_token_distribution(model, tokenizer, prompts):
    """Analyze what tokens the model assigns high probability to."""
    print("\n" + "="*70)
    print("🧪 TEST 4: Token Probability Distribution")
    print("="*70)
    
    device = next(model.parameters()).device
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits[0, -1, :], dim=-1)
        
        # Entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        # Top tokens
        top_probs, top_ids = torch.topk(probs, 10)
        
        print(f"\nPrompt: '{prompt}'")
        print(f"  Entropy: {entropy:.2f} (higher = more uncertain)")
        print(f"  Top probability: {top_probs[0].item():.4f}")
        print("  Top 10 tokens:")
        for prob, idx in zip(top_probs, top_ids):
            token = tokenizer.decode([idx.item()])
            print(f"    '{token}' ({prob.item():.4f})")

def check_tokenizer_quality(tokenizer):
    """Check if tokenizer produces sensible tokens."""
    print("\n" + "="*70)
    print("🧪 TEST 5: Tokenizer Quality Check")
    print("="*70)
    
    test_texts = [
        "Ciao, come stai oggi?",
        "La fotosintesi è un processo biologico.",
        "The quick brown fox jumps over the lazy dog.",
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        token_strs = [tokenizer.decode([t]) for t in tokens]
        
        print(f"\nOriginal: '{text}'")
        print(f"Decoded:  '{decoded}'")
        print(f"Tokens ({len(tokens)}): {token_strs[:15]}...")
        
        # Check for byte-level garbage
        has_weird = any(len(t) == 1 and ord(t[0]) > 127 for t in token_strs)
        if has_weird:
            print("  ⚠️ Warning: Contains byte-level tokens (potential encoding issue)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/elleci_step_5000.pth")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_chimera_v2_patched")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Run all tests
    test_generation_params(model, tokenizer)
    analyze_lm_head_weights(model, tokenizer)
    analyze_token_distribution(model, tokenizer, ["Ciao", "The", "La capitale d'Italia è"])
    check_tokenizer_quality(tokenizer)
    inspect_dataset_samples(tokenizer, n_samples=5)
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
