"""
Elleci V1 - Checkpoint Diagnostic
Analyze what's wrong with a checkpoint.
"""
import torch
import numpy as np
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

def test_logits_distribution(model, tokenizer, device):
    """Check if logits are healthy."""
    print("\n" + "="*60)
    print("📊 TEST 1: Logits Distribution")
    print("="*60)
    
    test_inputs = [
        "Ciao",
        "The quick brown fox",
        "La fotosintesi è un processo",
    ]
    
    for text in test_inputs:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        
        # Stats
        print(f"\nInput: '{text}'")
        print(f"  Logits - min: {last_logits.min():.2f}, max: {last_logits.max():.2f}, mean: {last_logits.mean():.2f}, std: {last_logits.std():.2f}")
        print(f"  Probs - max: {probs.max():.4f}, entropy: {-(probs * torch.log(probs + 1e-10)).sum():.2f}")
        
        # Top 10 tokens
        top_probs, top_ids = torch.topk(probs, 10)
        print(f"  Top 10 tokens:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_ids)):
            token = tokenizer.decode([idx.item()])
            print(f"    {i+1}. '{token}' ({prob:.4f})")

def test_embedding_health(model):
    """Check embedding weights."""
    print("\n" + "="*60)
    print("📊 TEST 2: Embedding Health")
    print("="*60)
    
    emb = model.token_emb.weight.data
    print(f"Token embeddings shape: {emb.shape}")
    print(f"  Mean: {emb.mean():.6f}")
    print(f"  Std: {emb.std():.6f}")
    print(f"  Min: {emb.min():.6f}")
    print(f"  Max: {emb.max():.6f}")
    print(f"  % zeros: {(emb == 0).sum().item() / emb.numel() * 100:.2f}%")
    print(f"  % NaN: {torch.isnan(emb).sum().item() / emb.numel() * 100:.2f}%")
    
    # Check if embeddings are collapsed
    emb_norms = emb.norm(dim=1)
    print(f"  Embedding norms - mean: {emb_norms.mean():.4f}, std: {emb_norms.std():.4f}")
    
    # Check similarity between random embeddings
    idx1, idx2 = torch.randint(0, emb.shape[0], (2,))
    sim = torch.cosine_similarity(emb[idx1:idx1+1], emb[idx2:idx2+1])
    print(f"  Random pair cosine similarity: {sim.item():.4f}")

def test_layer_outputs(model, tokenizer, device):
    """Check intermediate layer outputs."""
    print("\n" + "="*60)
    print("📊 TEST 3: Layer Output Analysis")
    print("="*60)
    
    input_ids = tokenizer.encode("Ciao, come stai?", return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get embedding output (RoPE provides position info in attention layers)
        x = model.token_emb(input_ids)
        print(f"After embedding: mean={x.mean():.4f}, std={x.std():.4f}")
        
        # Check first few blocks
        for i, block in enumerate(model.blocks[:3]):
            x = block(x)
            print(f"After block {i}: mean={x.mean():.4f}, std={x.std():.4f}, max={x.abs().max():.4f}")

def test_greedy_generation(model, tokenizer, device):
    """Test greedy vs sampling generation."""
    print("\n" + "="*60)
    print("📊 TEST 4: Greedy Generation (temp=0)")
    print("="*60)
    
    prompts = ["Ciao", "La capitale d'Italia è", "What is"]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = []
        
        with torch.no_grad():
            for _ in range(20):
                outputs = model(input_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                next_token = logits[0, -1, :].argmax()
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        generated_text = tokenizer.decode(generated)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
        print()

def test_perplexity(model, tokenizer, device):
    """Calculate perplexity on sample text."""
    print("\n" + "="*60)
    print("📊 TEST 5: Perplexity Check")
    print("="*60)
    
    test_texts = [
        "La fotosintesi è un processo biologico fondamentale.",
        "The quick brown fox jumps over the lazy dog.",
        "aaaaaaaaaaaaaaaaaaaa",  # Should have HIGH perplexity
        ".,.,.,.,.,.,.,.,.,.,",  # Punctuation pattern
    ]
    
    for text in test_texts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
        # Shift for loss calculation
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        perplexity = torch.exp(loss)
        
        print(f"Text: '{text[:40]}...' -> PPL: {perplexity.item():.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/elleci_step_10000.pth")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_chimera_v2_patched")
    model, config = load_model(args.checkpoint, device)
    
    # Run all tests
    test_embedding_health(model)
    test_logits_distribution(model, tokenizer, device)
    test_layer_outputs(model, tokenizer, device)
    test_greedy_generation(model, tokenizer, device)
    test_perplexity(model, tokenizer, device)
    
    print("\n" + "="*60)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
