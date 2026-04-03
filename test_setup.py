import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

api_key = os.getenv("LLAMA_CLOUD_API_KEY")

if api_key and api_key.startswith("llx-"):
    print("✅ LlamaCloud API Key loaded successfully!")
else:
    print("❌ API Key not found. Check your .env file.")