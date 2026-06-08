import json
from rank_bm25 import BM25Okapi
from app.storage.redis_cache import redis_client

SPARSE_KEY = "sparse:chunks"

# In-memory cache of current session's BM25 index
_BM25   = None
_CHUNKS = []


def _load_from_redis():
    """Load all chunks stored in Redis into memory for BM25."""
    global _BM25, _CHUNKS

    data = redis_client.get(SPARSE_KEY)
    if not data:
        _BM25   = None
        _CHUNKS = []
        return

    _CHUNKS   = json.loads(data)
    tokenized = [c["text"].split() for c in _CHUNKS]
    _BM25     = BM25Okapi(tokenized)


def add_to_sparse(new_chunks: list[dict]):
    """Add new chunks to the persistent BM25 store in Redis."""
    global _BM25, _CHUNKS

    existing_data = redis_client.get(SPARSE_KEY)
    existing      = json.loads(existing_data) if existing_data else []

    # Deduplicate by chunk_id
    existing_ids = {c["chunk_id"] for c in existing}
    added        = [c for c in new_chunks if c["chunk_id"] not in existing_ids]

    if not added:
        return

    all_chunks = existing + added
    redis_client.set(SPARSE_KEY, json.dumps(all_chunks))

    _CHUNKS   = all_chunks
    tokenized = [c["text"].split() for c in _CHUNKS]
    _BM25     = BM25Okapi(tokenized)


def sparse_search(query: str, k: int = 8, filter_source=None) -> list[dict]:
    global _BM25, _CHUNKS

    # Lazy load from Redis if not in memory
    if _BM25 is None:
        _load_from_redis()

    if not _BM25:
        return []

    scores = _BM25.get_scores(query.split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked:
        chunk = _CHUNKS[i]

        if filter_source and chunk.get("source") != filter_source:
            continue

        results.append(chunk)

        if len(results) >= k:
            break

    return results


def get_sparse_chunk_count() -> int:
    data = redis_client.get(SPARSE_KEY)
    if not data:
        return 0
    return len(json.loads(data))
