import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "hybrid-rag-advanced")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Models
LLM_MODEL        = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Chunking
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K       = 3
CHROMA_PATH = "chroma_db"

# Reranker
RERANK_MODEL = "BAAI/bge-reranker-base"
