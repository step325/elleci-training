
import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ API Key not found")
    exit(1)

client = genai.Client(api_key=api_key)

print("🔍 Listing available models (google-genai SDK)...")
try:
    # Method 1: direct list
    for m in client.models.list():
        print(f"- {m.name} | {m.display_name}")
        
except Exception as e:
    print(f"❌ Error listing models: {e}")
