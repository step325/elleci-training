"""
Elleci v2 - Terminal Chat Interface
Features:
- Maintains conversation history
- System prompts
- Colored output
- Infinite loop with command handling
"""
import torch
import sys
import os
from pathlib import Path
from transformers import AutoTokenizer

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ElleciConfig
from src.model import Elleci

# ANSI Colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

class ChatSession:
    def __init__(self, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"{YELLOW}Loading Elleci v2 from {checkpoint_path}...{RESET}")
        self.config = ElleciConfig()
        self.config.n_layers = 6
        self.config.max_seq_len = 512 # Extended for chat history
        self.config.device = device
        
        self.model = Elleci(self.config)
        
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"{BLUE}[Warning] Checkpoint not found. Initializing random weights for testing.{RESET}")
            
        self.model.to(device)
        self.model.eval()
        
        self.history = ""
        self.max_history_tokens = 400 # Keep room for generation
        
        print(f"{GREEN}✓ Model Ready.{RESET}")

    def generate_response(self, user_input, temperature=0.7, top_k=50):
        # 1. Format Input
        # Simple chat format: "User: <input>\nAI:"
        new_entry = f"User: {user_input}\nAI:"
        
        # 2. Add to history
        # (In a real scenario, we'd manage a sliding window here)
        current_context = self.history + new_entry
        
        # 3. Tokenize
        tokens = self.tokenizer.encode(current_context, add_special_tokens=False)
        
        # 4. Truncate if too long (keep latest)
        if len(tokens) > self.max_history_tokens:
            tokens = tokens[-self.max_history_tokens:]
            # Re-decode to ensure valid text boundary (optional but safer)
            current_context = self.tokenizer.decode(tokens)
            
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # 5. Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=100, # Response length
                temperature=temperature,
                top_k=top_k
            )
            
        # 6. Decode ONLY the new part
        generated_ids = output_ids[0][len(tokens):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Stop at "User:" if the model hallucinates the user turn
        if "User:" in response:
            response = response.split("User:")[0]
            
        # 7. Update History
        self.history += new_entry + " " + response + "\n"
        
        return response

def start_chat():
    checkpoint = "nanoprime_v2_final.pth"
    session = ChatSession(checkpoint)
    
    print("-" * 50)
    print(f"{BOLD}Elleci v2 CLI Chat{RESET}")
    print("Commands: /reset, /quit, /temp <0.1-2.0>")
    print("-" * 50)
    
    current_temp = 0.7
    
    while True:
        try:
            user_input = input(f"\n{BLUE}You:{RESET} ").strip()
        except EOFError:
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ['/quit', '/exit']:
            break
            
        if user_input.lower() == '/reset':
            session.history = ""
            print(f"{YELLOW}Context cleared.{RESET}")
            continue
            
        if user_input.lower().startswith('/temp'):
            try:
                val = float(user_input.split()[1])
                current_temp = max(0.1, min(2.0, val))
                print(f"{YELLOW}Temperature set to {current_temp}{RESET}")
            except:
                print(f"{YELLOW}Invalid format. Use /temp 0.8{RESET}")
            continue
            
        # Generate
        print(f"{GREEN}Elleci:{RESET}", end=" ", flush=True)
        response = session.generate_response(user_input, temperature=current_temp)
        print(response)

if __name__ == "__main__":
    start_chat()
