"""
Project Chimera - Italian-Optimized BPE Tokenizer Training
Trains a 32k vocab tokenizer with 70% Italian / 30% English mix.
Forces common Italian words to be single tokens via higher frequency exposure.

Usage:
    python scripts/train_tokenizer_chimera.py
"""
import os
import random
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

# Configuration
VOCAB_SIZE = 32000
SAMPLE_SIZE = 2_000_000
OUTPUT_DIR = "tokenizer_chimera_v2"

# Italian words to boost (repeated in corpus to ensure single-token)
ITALIAN_BOOST_WORDS = [
    "ciao", "buongiorno", "buonasera", "buonanotte", "arrivederci", "salve",
    "grazie", "prego", "scusa", "scusi", "permesso",
    "questo", "questa", "quello", "quella", "qualcosa", "qualcuno",
    "essere", "avere", "fare", "dire", "andare", "venire", "vedere", "sapere",
    "potere", "volere", "dovere", "stare", "dare", "parlare", "pensare",
    "perche", "quindi", "pero", "mentre", "quando", "dove", "come",
    "anche", "ancora", "sempre", "spesso", "insieme", "inoltre",
    "proprio", "davvero", "veramente", "assolutamente", "probabilmente",
    "cosa", "tempo", "modo", "mondo", "anno", "giorno", "volta", "parte",
    "esempio", "problema", "lavoro", "paese", "citta", "casa", "vita",
    "quanto", "quanti", "quale", "quali", "chi", "benissimo", "perfetto",
]


def get_training_corpus():
    """
    Generator yielding text from mixed corpus.
    Strategy: 30% English, 55% Italian Wikipedia, 15% Italian Chat/Instructions.
    """
    print("Loading datasets...")
    
    # English: SmolLM-Corpus
    ds_en = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 
                         split="train", streaming=True)
    iter_en = iter(ds_en)
    
    # Italian: Wikipedia
    ds_it = load_dataset("wikipedia", "20220301.it", 
                         split="train", streaming=True, trust_remote_code=True)
    iter_it = iter(ds_it)
    
    # Italian Chat: OASST1 (filtered for Italian)
    try:
        ds_chat = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
        iter_chat = iter(ds_chat)
    except Exception as e:
        print(f"Warning: Could not load OASST1: {e}")
        iter_chat = iter([])
    
    # Local Italian instructions if available
    local_instructions = []
    local_path = "data/native_instructions_v1.jsonl"
    if os.path.exists(local_path):
        import json
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = item.get("instruction", "") + " " + item.get("output", "")
                    if text.strip():
                        local_instructions.append(text)
                except:
                    pass
        print(f"Loaded {len(local_instructions)} local Italian instructions")
    
    # Boost corpus: repeat important Italian words many times
    boost_text = " ".join(ITALIAN_BOOST_WORDS * 500)
    
    print(f"Streaming {SAMPLE_SIZE} samples (30% En, 55% It Wiki, 15% It Chat)...")
    
    # Yield boost words first to ensure they become tokens
    for _ in range(100):
        yield boost_text
    
    for i in range(SAMPLE_SIZE):
        r = random.random()
        
        if r < 0.30:
            # English (30%)
            try:
                item = next(iter_en)
                yield item.get("text", "")
            except StopIteration:
                pass
                
        elif r < 0.85:
            # Italian Wikipedia (55%)
            try:
                item = next(iter_it)
                yield item.get("text", "")
            except StopIteration:
                pass
                
        else:
            # Italian Chat/Instructions (15%)
            if local_instructions and random.random() < 0.5:
                yield random.choice(local_instructions)
            else:
                try:
                    item = next(iter_chat)
                    if item.get("lang") == "it":
                        yield item.get("text", "")
                except StopIteration:
                    pass


def train_tokenizer():
    # Initialize BPE
    tokenizer = Tokenizer(models.BPE())
    
    # GPT-2 style pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Trainer with special tokens
    special_toks = ["<|endoftext|>", "<|padding|>", "<|im_start|>", "<|im_end|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_toks,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Train
    print("Starting BPE Training (this will take several minutes)...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # Post-processing
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Wrap in Transformers format
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|padding|>",
        unk_token="<|endoftext|>"
    )
    
    fast_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Tokenizer saved to '{OUTPUT_DIR}'")
    
    return fast_tokenizer


def validate_tokenizer(tokenizer):
    print("\nValidation: Checking Italian Efficiency")
    test_words = [
        "ciao",
        "buongiorno", 
        "grazie mille",
        "come stai",
        "questo problema",
        "dell'intelligenza",
        "perche no",
    ]
    
    for word in test_words:
        tokens = tokenizer.encode(word)
        tokens_str = [tokenizer.decode([t]) for t in tokens]
        print(f"  '{word}' -> {len(tokens)} tokens: {tokens_str}")
        
    print("\nValidation Complete. Lower token counts = better!")


if __name__ == "__main__":
    trained_tok = train_tokenizer()
    validate_tokenizer(trained_tok)
