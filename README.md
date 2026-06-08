# Hallucination-Resistant AI Search Assistant

A local RAG system built with LangGraph and LangChain. Supports file uploads, on-demand web ingestion, hybrid retrieval, a self-critiquing agent loop, and Redis-backed caching. Runs fully offline via Ollama with no OpenAI API key required.

---

## What it does

- Routes queries across 7 specialised agents (RAG, web search, SQL, coding, math, general knowledge, file context) using an embedding-based semantic classifier
- Retrieves from a hybrid dense + sparse index (ChromaDB + BM25) fused via Reciprocal Rank Fusion, then reranked with a cross-encoder
- Rewrites and expands queries before retrieval for better recall
- Runs a Self-RAG critic loop that checks if answers are grounded and retries retrieval if not (up to 2 retries)
- Ingests files (PDF, DOCX, TXT, MD) and fetches web content on demand when local knowledge is insufficient
- Caches query results in Redis with a 1-hour TTL; BM25 index also persists in Redis across restarts
- Checks inputs through keyword and LLM-based guardrails before any processing
- Traces agent runs through LangSmith
- Shows a live debug sidebar with cache stats, indexed sources, and latency

---

## Tech Stack

| Layer | Tools |
|---|---|
| LLM | Ollama (qwen2.5:14b) |
| Embeddings | Ollama (nomic-embed-text) |
| Vector Store | ChromaDB |
| Sparse Retrieval | BM25 (rank-bm25), persisted in Redis |
| Reranker | sentence-transformers (BAAI/bge-reranker-base) |
| Agent Orchestration | LangGraph |
| Web Ingestion | DuckDuckGo Search + BeautifulSoup4 |
| File Ingestion | pypdf, python-docx |
| Caching | Redis |
| Observability | LangSmith |
| UI | Streamlit |

---

## Project Structure

```
app/
├── main.py                      # Streamlit UI
├── config.py                    # Config and env vars
├── agent_builder.py             # Builds the LangChain agent
├── tools.py                     # rag_search tool
├── agent/
│   ├── graph.py                 # LangGraph state graph
│   ├── nodes.py                 # Retriever, Generator, Critic nodes
│   └── state.py                 # AgentState schema
├── agents/
│   ├── classifier.py            # Embedding-based query classifier
│   ├── router.py                # Routes queries to the right agent
│   ├── rag_agent.py
│   ├── web_search_agent.py
│   ├── coding_agent.py
│   ├── sql_agent.py
│   ├── math_agent.py
│   └── general_knowledge_agent.py
├── retrieval/
│   ├── pipeline.py              # Hybrid RRF + rerank pipeline
│   ├── dense.py                 # ChromaDB search
│   ├── sparse.py                # BM25 search
│   └── reranker.py              # Cross-encoder reranking
├── ingestion/
│   ├── file_ingest.py           # File ingestion
│   ├── pipeline.py              # Web ingestion pipeline
│   ├── search.py                # DuckDuckGo search
│   └── chunking.py              # Text chunking
├── indexing/
│   └── indexer.py               # Indexes chunks into ChromaDB and BM25
├── storage/
│   ├── document_store.py        # Document dedup and storage
│   └── redis_cache.py           # Redis query cache
├── query/
│   └── rewrite.py               # Query rewriting and expansion
├── guardrails/
│   └── safety.py                # Input safety checks
└── logging/
    ├── logger.py
    └── evaluator.py             # Response quality evaluator
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- Redis running locally or via Docker
- A [LangSmith](https://smith.langchain.com) account (free) for tracing

### Steps

**1. Clone the repo**
```bash
git clone https://github.com/your-username/RAG_Chatbot.git
cd RAG_Chatbot
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Pull Ollama models**
```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

**5. Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env`:
```env
LANGCHAIN_API_KEY=your_langsmith_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
```

**6. Start Redis**
```bash
# Docker
docker run -d -p 6379:6379 redis

# Or locally
redis-server
```

**7. Run the app**
```bash
python -m streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Notes

- `chroma_db/` is created automatically on first run and is gitignored
- BM25 index persists in Redis under `sparse:chunks` and survives app restarts
- Duplicate files are detected by content hash and skipped
- LangSmith tracing is optional but recommended for inspecting the agent graph
