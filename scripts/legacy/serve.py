from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
import uvicorn

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.config import NanoPrimeConfig
from src.model import NanoPrime

app = FastAPI(title="NanoPrime v2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
MODEL = None
TOKENIZER = None
CONFIG = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50

@app.on_event("startup")
async def load_model():
    global MODEL, TOKENIZER, CONFIG
    print("🚀 Loading NanoPrime v2...")
    
    checkpoint_path = "nanoprime_v2_italian.pth"
    TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    TOKENIZER.pad_token = TOKENIZER.eos_token
    
    CONFIG = NanoPrimeConfig()
    CONFIG.n_layers = 6
    CONFIG.max_seq_len = 128 # Must match training (Cosmopedia/Instruction)
    CONFIG.device = DEVICE
    
    MODEL = NanoPrime(CONFIG)
    
    # Load weights if available
    if Path(checkpoint_path).exists():
        try:
            # Try loading with strict=False to ignore extra keys (like Router layers) if compatible
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            MODEL.load_state_dict(state_dict, strict=False)
            print("✅ Weights loaded (Note: strict=False used, ignored keys or mismatches handled).")
        except Exception as e:
            print(f"⚠️ Checkpoint Mismatch: {e}")
            print("⚠️ The found checkpoint is likely an old test or different architecture.")
            print("⚠️ using RANDOM WEIGHTS for testing infrastructure.")
    else:
        print("⚠️ Checkpoint not found. Using random weights.")
        
    MODEL.to(DEVICE)
    MODEL.eval()
    print("✅ System Ready.")

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/v1/generate")
async def generate(req: GenerateRequest):
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Tokenize
        tokens = TOKENIZER.encode(req.prompt, add_special_tokens=False)
        # Truncate
        if len(tokens) > 400:
            tokens = tokens[-400:]
            
        input_ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            output = MODEL.generate(
                input_ids, 
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k
            )
            
        # Decode only new tokens
        new_tokens = output[0][len(tokens):]
        text = TOKENIZER.decode(new_tokens, skip_special_tokens=True)
        
        return {"text": text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
