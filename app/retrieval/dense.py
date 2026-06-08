"""
Dense retrieval via ChromaDB.
"""

import os
import chromadb
from app.logging.logger import logger
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from app.config import EMBED_MODEL, CHROMA_PATH

_client     = None
_collection = None
_embeddings = OllamaEmbeddings(model=EMBED_MODEL)

def _get_collection():
    global _client, _collection
    if _collection is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

        _collection = _client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# Write
def add_to_db(chunks: list[dict]):
    col = _get_collection()

    # Deduplicate against what's already in the collection
    existing_ids = set(col.get(ids=[c["chunk_id"] for c in chunks])["ids"])
    new_chunks   = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        return

    texts      = [c["text"]     for c in new_chunks]
    ids        = [c["chunk_id"] for c in new_chunks]
    metadatas  = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in new_chunks]
    embeddings = _embeddings.embed_documents(texts)

    col.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)


# Read
def dense_search(query: str, k: int = 8, filter_source=None) -> list[dict]:
    col = _get_collection()

    if col.count() == 0:
        return []

    query_embedding = _embeddings.embed_query(query)

    # Build where clause and only apply if filter_source is set
    # Use None (no filter) rather than risking a failed where on missing metadata
    where = {"source": {"$eq": filter_source}} if filter_source else None

    try:
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, col.count()),
            where=where,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        # If filtered query fails (no chunks with that source), retry without filter
        logger.warning(f"DENSE SEARCH filtered query failed ({e}), retrying without filter")
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, col.count()),
            include=["documents", "metadatas"],
        )

    docs      = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Post-filter by source if chroma where clause was skipped on retry
    rows = [
        {"text": doc, "source": meta.get("source"), "chunk_id": meta.get("chunk_id")}
        for doc, meta in zip(docs, metadatas)
    ]
    if filter_source:
        rows = [r for r in rows if r["source"] == filter_source]

    return rows

def fetch_all_chunks_for_source(source: str) -> list[dict]:
    """Return every chunk stored for a given source, ordered by chunk_id."""
    try:
        col = _get_collection()
        results = col.get(
            where={"source": {"$eq": source}},
            include=["documents", "metadatas"],
        )
        rows = [
            {"text": doc, "source": meta.get("source"), "chunk_id": meta.get("chunk_id")}
            for doc, meta in zip(results.get("documents", []), results.get("metadatas", []))
        ]
        return sorted(rows, key=lambda r: r["chunk_id"])
    except Exception as e:
        logger.warning(f"fetch_all_chunks_for_source failed: {e}")
        return []


def source_exists_in_chroma(source: str) -> bool:
    """
    Check whether at least one chunk exists
    for a given source.
    """
    try:
        col = _get_collection()

        results = col.get(
            where={"source": source},
            limit=1,
        )

        return bool(results.get("ids"))

    except Exception:
        return False
    
# Stats
def get_chroma_stats() -> dict:
    try:
        col     = _get_collection()
        count   = col.count()
        sources = set()

        if count > 0:
            results = col.get(include=["metadatas"])
            for m in results.get("metadatas", []):
                if m and m.get("source"):
                    sources.add(m["source"])

        return {
            "total_chunks":    count,
            "unique_sources":  len(sources),
            "sources":         list(sources),
            "path":            os.path.abspath(CHROMA_PATH),
        }
    except Exception as e:
        return {"error": str(e), "total_chunks": 0, "unique_sources": 0, "sources": []}