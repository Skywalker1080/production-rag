"""Central config loaded from .env. Keep it simple."""
import os

from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Auth option 1 (you have this): Bedrock API key / bearer token.
# In .env set: AWS_BEARER_TOKEN_BEDROCK=<your bedrockapi key>
AWS_BEARER_TOKEN_BEDROCK = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")

# Auth option 2: classic IAM keys (optional, not needed if bearer token set).
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "zai.glm-5")
BEDROCK_EMBED_MODEL_ID = os.getenv(
    "BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"
)

# --- Embeddings provider: cohere (Bedrock, default) | gemini | ollama | bedrock ---
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "cohere").lower()
COHERE_EMBED_MODEL_ID = os.getenv("COHERE_EMBED_MODEL_ID", "cohere.embed-english-v3")
COHERE_BATCH_SIZE = int(os.getenv("COHERE_BATCH_SIZE", "96"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_EMBED_DIM = int(os.getenv("OLLAMA_EMBED_DIM", "768"))
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
GEMINI_EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "1536"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_docs")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "8"))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))
EMBED_PACING_SECS = float(os.getenv("EMBED_PACING_SECS", "2"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "200"))

DATA_DIR = os.getenv("DATA_DIR", "./data")

# --- Logging ---
LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_FILE = os.getenv("LOG_FILE", "./logs/rag.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- PDF loader (unstructured, layout-aware for multi-column PDFs) ---
# fast (default) = text extraction, seconds, no downloads, weaker column order.
# hi_res = layout model, minutes on CPU, best multi-column reading order
# (needs poppler + tesseract, see README). auto picks hi_res when possible.
UNSTRUCTURED_STRATEGY = os.getenv("UNSTRUCTURED_STRATEGY", "fast")
