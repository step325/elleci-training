"""
Generate Synthetic Textbook Data (OpenRouter Exclusive)
=======================================================

This script produces "Textbook Quality" training data using Top-Tier models 
available via OpenRouter's Free Tier (Supporter Level).

Features:
- **Best-in-Class Models**: Llama 3.1 405B, DeepSeek R1, Gemini 2.0 Flash Exp.
- **Rotation Strategy**: Distributes load across models to avoid provider-specific rate limits.
- **Infinite Mode**: Continuous generation with queue management.

Prerequisites:
    pip install openai tqdm

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python scripts/data_quality/generate_free_data.py --infinite
"""

import os
import sys
import argparse
import time
import json
import random
from datetime import datetime
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("❌ Library 'openai' not found. Please install: pip install openai")
    sys.exit(1)

# Configuration
OUTPUT_FILE = "data/synthetic_textbooks_it_free.jsonl"
DEFAULT_DELAY = 10  # Seconds between requests (Safe pace for 1000 RPD over ~3h sessions)

# Prompt Template (Phi-style / Textbook Quality)
SYSTEM_PROMPT = """
Sei un professore universitario esperto e un autore di best-seller di manuali scolastici.
Il tuo compito è scrivere capitoli di un libro di testo di ALTISSIMA qualità (Textbook Quality) in Italiano.

REGOLE DI SCRITTURA:
1.  **Stile**: Didattico, denso di informazioni, chiaro, coinvolgente.
2.  **Struttura**:
    *   ## Titolo del Concetto
    *   **Spiegazione Teorica**: Profonda ma accessibile.
    *   **Esempi Pratici**: Codice (se coding), Formule (se fisica/mate), Analogie (se storia/filosofia).
    *   **Esercizi Svolti**: Almeno uno passo-passo.
3.  **Formattazione**: Usa Markdown (grassetto, liste, blocchi di codice).
4.  **Lingua**: Italiano perfetto.

NON scrivere introduzioni tipo "Ecco il capitolo". Scrivi DIRETTAMENTE il contenuto del libro.
"""

# START SELECTION: Best Free Models (Jan 2026)
# Criteria: High Params (Smart), High Context, Reasoning Capabilities
MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",  # ⚡ Fast & Smart (Primary)
    "google/gemma-3-27b-it:free",              # 🌍 Great Multilingual
    "deepseek/deepseek-r1-0528:free",          # 🧠 SOTA Reasoning      # 📚 1M Context
    "qwen/qwen-2.5-vl-7b-instruct:free",       # 👁️ Visual/Dense
    "mistralai/mistral-7b-instruct:free",      # 🐎 Workhorse fallback
    "nvidia/nemotron-3-nano-30b-a3b:free"      # 🎮 Nvidia Optimized
]

class Generator:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_index = 0
        self.banned_models = set()

    def get_next_model(self):
        """Rotate through the model list (skipping banned ones)."""
        # Filter active models
        active_models = [m for m in MODELS if m not in self.banned_models]
        
        if not active_models:
            print("❌ All models banned! Resetting ban list.")
            self.banned_models.clear()
            active_models = MODELS
            
        model = active_models[self.model_index % len(active_models)]
        self.model_index = (self.model_index + 1) % len(active_models) # Keep index moving
        return model

    def generate_content(self, topic):
        """Generate content attempting multiple models if one fails."""
        
        full_prompt = f"Scrivi un capitolo completo su: {topic}.\nConcentrati su aspetti avanzati, sfumature, esempi concreti e casi d'uso reali."
        
        retries = len(MODELS) # Try all unique models if needed
        for _ in range(retries):
            model = self.get_next_model()
            try:
                print(f"   🤖 Model: {model.split('/')[-1]} ... ", end="")
                sys.stdout.flush()
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=8192,
                    extra_headers={
                        "HTTP-Referer": "https://elleci.ai",
                        "X-Title": "Elleci DataGen"
                    }
                )
                
                content = response.choices[0].message.content
                if content and len(content) > 100:
                    print(f"✅ OK ({len(content)} chars)")
                    return content, model
                else:
                    print("⚠️ Empty? Trying next.")
            
            except Exception as e:
                err_str = str(e)
                print(f"❌ Error: {err_str[:100]}...") # Truncate log
                
                # Auto-Ban logic for fatal errors (404, 400, etc)
                if "404" in err_str or "Route not found" in err_str or "not configured" in err_str:
                    print(f"   🚫 Blacklisting {model} (Route Error)")
                    self.banned_models.add(model)
                
                time.sleep(1) # Short penalty
        
        return None, None

    def brainstorm_topics(self, recent_topics):
        """Ask the smartest model currently available for new topics."""
        prompt = f"""
        Role: Academic Editor.
        Task: List 10 NEW, ADVANCED, SPECIFIC textbook chapter titles (Physics, History, CS, Philo, Bio).
        Avoid these recent ones: {json.dumps(recent_topics[-5:], ensure_ascii=False)}
        Format: Just the titles, one per line. No bullets.
        """
        
        # Prefer smart models for brainstorming
        smart_models = [m for m in MODELS if "405" in m or "r1" in m or "70" in m]
        model = random.choice(smart_models)
        
        try:
            print(f"\n🧠 Brainstorming with {model.split('/')[-1]}...")
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9
            )
            text = response.choices[0].message.content
            lines = [l.strip().replace("* ", "").replace("- ", "").replace('"', "") for l in text.splitlines() if l.strip()]
            return lines[:10]
        except Exception as e:
            print(f"⚠️ Brainstorming failed: {e}")
            return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infinite", action="store_true", help="Run forever")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found.")
        print("👉 Please run: export OPENROUTER_API_KEY='sk-or-...'")
        sys.exit(1)

    generator = Generator(api_key)
    
    # State Management
    queue = [
        "Fisica: La Teoria delle Stringhe e le dimensioni extra",
        "Storia: L'impatto della Peste Nera sull'economia medievale",
        "Coding: Architetture a Microservizi vs Monoliti",
        "Filosofia: L'Imperativo Categorico di Kant oggi"
    ]
    seen_topics = set()

    # Resume capability
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "topic" in d: seen_topics.add(d["topic"])
                    except: pass
            print(f"📚 Resuming: {len(seen_topics)} chapters already generated.")
        except Exception as e:
            print(f"⚠️ Error reading history: {e}")

    # Main Loop
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"🚀 Starting OpenRouter Factory (Supporter Mode)")
    print(f"💾 Output: {OUTPUT_FILE}")
    print("-" * 60)

    try:
        while True:
            # 1. Refill Queue
            if len(queue) < 3:
                new_topics = generator.brainstorm_topics(list(seen_topics)[-10:])
                for t in new_topics:
                    if t not in seen_topics and t not in queue:
                        queue.append(t)
                
                if not queue: # Fallback if brainstorming fails
                    queue.append(f"Argomento Avanzato {len(seen_topics)+1}")
            
            # 2. Pick Topic
            if not queue: break
            topic = queue.pop(0)
            if topic in seen_topics: continue

            print(f"\n✍️  Topic: {topic}")

            # 3. Generate
            content, used_model = generator.generate_content(topic)

            # 4. Save
            if content:
                entry = {
                    "source": used_model,
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                    "text": content,
                    "provider": "openrouter"
                }
                
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
                seen_topics.add(topic)
                
                # 5. Rate Limit Safety (Supporter Tier)
                # 1000 RPD = ~0.7 RPM (continuous) or ~5-10 RPM (burst).
                # We use a safe comfortable pace.
                sleep_time = random.uniform(8, 12)
                print(f"💤 Cooling down ({sleep_time:.1f}s)...")
                time.sleep(sleep_time)
            
            if not args.infinite and not queue:
                break

    except KeyboardInterrupt:
        print("\n🛑 Factory stopped by user.")

    print(f"\n✅ Done. Total unique chapters: {len(seen_topics)}")

if __name__ == "__main__":
    main()
