"""
Project Chimera - Add Italian Tokens to Tokenizer
Adds common Italian words as dedicated tokens to prevent splitting.
This must be run AFTER BPE training.

Usage:
    python scripts/add_italian_tokens.py
"""
import os
from transformers import PreTrainedTokenizerFast, AddedToken

# Input/Output
INPUT_TOKENIZER = "tokenizer_chimera_v2"
OUTPUT_TOKENIZER = "tokenizer_chimera_v2_patched"

# Italian words that MUST be single tokens
ITALIAN_TOKENS = [
    # Greetings
    "ciao", "buongiorno", "buonasera", "buonanotte", "arrivederci", "salve",
    # Courtesy
    "grazie", "prego", "scusa", "scusi", "permesso",
    # Pronouns
    "questo", "questa", "quello", "quella", "qualcosa", "qualcuno",
    # Common verbs (infinitive forms)
    "essere", "avere", "fare", "dire", "andare", "venire", "potere", "volere",
    "dovere", "sapere", "vedere", "stare", "dare", "parlare", "pensare",
    # Conjunctions & connectors
    "quindi", "mentre", "quando", "perche", "anche", "ancora", "sempre",
    "proprio", "davvero", "veramente",
    # Common nouns
    "esempio", "problema", "giorno", "lavoro", "tempo", "mondo",
    # Expressions
    "benissimo", "perfetto", "esatto", "giusto", "certo",
]


def add_italian_tokens():
    print(f"Loading tokenizer from '{INPUT_TOKENIZER}'...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(INPUT_TOKENIZER)
    
    original_vocab_size = tokenizer.vocab_size
    print(f"Original vocab size: {original_vocab_size}")
    
    # Test before
    print("\n--- BEFORE adding tokens ---")
    test_words = ["ciao", "buongiorno", "grazie", "questo problema"]
    for word in test_words:
        tokens = tokenizer.encode(word)
        decoded = [tokenizer.decode([t]) for t in tokens]
        print(f"  '{word}' -> {len(tokens)} tokens: {decoded}")
    
    # Add tokens (as normal tokens, not special)
    new_tokens = [AddedToken(word, normalized=False) for word in ITALIAN_TOKENS if word not in tokenizer.get_vocab()]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"\nAdded {num_added} new tokens")
    print(f"New vocab size: {tokenizer.vocab_size}")
    
    # Test after
    print("\n--- AFTER adding tokens ---")
    for word in test_words:
        tokens = tokenizer.encode(word)
        decoded = [tokenizer.decode([t]) for t in tokens]
        print(f"  '{word}' -> {len(tokens)} tokens: {decoded}")
    
    # Save
    os.makedirs(OUTPUT_TOKENIZER, exist_ok=True)
    tokenizer.save_pretrained(OUTPUT_TOKENIZER)
    print(f"\nPatched tokenizer saved to '{OUTPUT_TOKENIZER}'")
    
    return tokenizer


if __name__ == "__main__":
    add_italian_tokens()
