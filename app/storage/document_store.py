import hashlib
import json

from app.storage.redis_cache import redis_client

DOCUMENT_PREFIX = "document:"
URL_PREFIX      = "url:"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def store_document(url: str, texts: list[str]):
    full_text = " ".join(texts).strip()

    if not full_text:
        return None

    doc_id        = _hash(full_text)
    document_key  = f"{DOCUMENT_PREFIX}{doc_id}"
    url_key       = f"{URL_PREFIX}{url}"

    # Deduplicate by content hash
    if redis_client.exists(document_key):
        redis_client.set(url_key, doc_id)
        return None

    redis_client.set(
        document_key,
        json.dumps({"doc_id": doc_id, "url": url, "text": full_text})
    )
    redis_client.set(url_key, doc_id)

    return doc_id


def get_document(doc_id: str):
    data = redis_client.get(f"{DOCUMENT_PREFIX}{doc_id}")
    if not data:
        return None
    return json.loads(data)


def seen_url(url: str) -> bool:
    return bool(redis_client.exists(f"{URL_PREFIX}{url}"))


def get_document_url(doc_id: str) -> str | None:
    """Reverse lookup: doc_id -> url by reading stored document metadata."""
    data = redis_client.get(f"{DOCUMENT_PREFIX}{doc_id}")
    if not data:
        return None
    return json.loads(data).get("url")


def list_all_documents():
    """Return all stored documents for the debug panel."""
    keys = redis_client.keys(f"{DOCUMENT_PREFIX}*")
    docs = []
    for key in keys:
        data = redis_client.get(key)
        if data:
            docs.append(json.loads(data))
    return docs
