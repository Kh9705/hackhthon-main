# config.py
import os
import torch
from dotenv import load_dotenv
from mistralai.async_client import MistralAsyncClient
from sentence_transformers import SentenceTransformer

# --- Load Environment Variables ---
load_dotenv()

# --- Mistral AI Configuration ---
API_KEYS = [key for key in os.environ if key.startswith("MISTRAL_API_KEY_")]
if not API_KEYS:
    print("[WARNING] No MISTRAL_API_KEY_<n> environment variables set!")
    MISTRAL_CREDENTIALS = []
else:
    MISTRAL_CREDENTIALS = [os.getenv(key) for key in API_KEYS]
    print(f"[INFO] Loaded {len(MISTRAL_CREDENTIALS)} Mistral AI keys.")

# Create a list of async Mistral clients
ASYNC_CLIENTS = [MistralAsyncClient(api_key=key) for key in MISTRAL_CREDENTIALS]
LLM_MODEL = "mistral-large-latest"

# --- Embedding Model Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
print(f"[INFO] Using device for embeddings: {DEVICE}")
print(f"[INFO] Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME}...")
ENCODER = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
print("[INFO] Embedding model loaded successfully.")

# --- Document Processing Configuration ---
SMALL_DOC_THRESHOLD = 20000
LARGE_DOC_THRESHOLD = 100000
CHUNK_CONFIG = {
    "small":  (500, 75, 10),
    "medium": (1000, 150, 7),
    "large":  (2000, 300, 5)
}