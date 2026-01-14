"""
Elleci v2 - Tokenizer Training Script 🗣️
Trains a custom BPE Tokenizer (32k vocab) on mixed English/Italian data.
Ratio: 60% English (SmolLM/FineWeb) / 40% Italian (CulturaX/Wiki).
"""
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import random

# Configuration
VOCAB_SIZE = 50304  # Upgraded to GPT-2 standard to fit Italian
SAMPLE_SIZE = 2_000_000 # Increased for better coverage
OUTPUT_DIR = "tokenizer_chimera_50k"

def get_training_corpus():
    """
    Generator that yields text from the mixed corpus.
    Strategy: 50% English reasoning, 50% Italian general (Balanced).
    """
    print("📚 Connecting to datasets...")
    
    # 1. English Source: SmolLM-Corpus (FineWeb-Edu subset)
    # Using 'cosmopedia-v2' subset as a proxy for high quality text if full fineweb is too heavy/slow to stream
    ds_en = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    iter_en = iter(ds_en)
    
    # 2. Italian Source: Wikipedia IT (Clean & Standard)
    ds_it = load_dataset("wikipedia", "20220301.it", split="train", streaming=True, trust_remote_code=True)
    iter_it = iter(ds_it)

    # 3. Conversational Source: OASST1 (Italian subset)
    # Adds "ciao", "grazie", "assistente" to vocab
    try:
        ds_chat = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
        # Filter for 'lang': 'it' manually during iteration
        iter_chat = iter(ds_chat)
    except Exception as e:
        print(f"⚠️ Warning: Could not load OASST1: {e}")
        iter_chat = iter([]) # Fallback
    
    print(f"🔄 Streaming & Mixing {SAMPLE_SIZE} samples (45% En, 45% It, 10% Chat)...")
    for _ in range(SAMPLE_SIZE):
        r = random.random()
        if r < 0.45:
            # English (45%)
            try:
                item = next(iter_en)
                yield item.get("text", "")
            except StopIteration:
                pass
        elif r < 0.90:
            # Italian Wikipedia (45%)
            try:
                item = next(iter_it)
                yield item.get("text", "")
            except StopIteration:
                pass
        else:
            # Italian Chat (10%)
            try:
                item = next(iter_chat)
                if item.get('lang') == 'it':
                    yield item.get("text", "")
                else:
                    # If not italian, skip or yield wiki fallback
                    pass 
            except StopIteration:
                pass

def train_tokenizer():
    # 1. Initialize BPE Tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # 2. Use GPT-2/4 style pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 3. Decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # 4. Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>", "<|padding|>", "<|im_start|>", "<|im_end|>"], # ChatML ready
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 5. Train
    print("🚀 Starting BPE Training (this might take a few minutes)...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # 6. Post-Processing
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # 7. Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Wrap in Transformers format for easy loading later
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|padding|>",
        unk_token="<|endoftext|>"
    )
    
    fast_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Tokenizer saved to '{OUTPUT_DIR}'")
    
    return fast_tokenizer

def validate_tokenizer(tokenizer):
    print("\n🔍 Validation: Checking Italian Efficiency")
    test_words = [
        "dell'intelligenza", 
        "perché", 
        "l'adattamento", 
        "precipitevolissimevolmente",
        "ciao come stai"
    ]
    
    for word in test_words:
        tokens = tokenizer.encode(word)
        tokens_str = [tokenizer.decode([t]) for t in tokens]
        print(f"  - '{word}' -> {len(tokens)} tokens: {tokens_str}")
        
    print("\n✅ Validation Complete. Check token counts above (Lower is better).")

if __name__ == "__main__":
    trained_tok = train_tokenizer()
    validate_tokenizer(trained_tok)
