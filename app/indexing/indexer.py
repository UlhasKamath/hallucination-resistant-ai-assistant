from app.storage.document_store import get_document, get_document_url
from app.ingestion.chunking import chunk_texts
from app.retrieval.dense import add_to_db, get_chroma_stats
from app.retrieval.sparse import add_to_sparse


def index_documents(doc_ids: list[str]):
    all_chunks = []

    for doc_id in doc_ids:
        doc = get_document(doc_id)
        if not doc:
            continue

        source_url = (
            doc.get("filename")
            or get_document_url(doc_id)
            or doc_id
        )
        text       = doc.get("text", "")

        if not text:
            continue

        chunks = chunk_texts([text])

        for i, c in enumerate(chunks):
            all_chunks.append({
                "text":     c,
                "source":   source_url or "uploaded",
                "chunk_id": f"{doc_id}_{i}"
            })

    if not all_chunks:
        return 0

    add_to_db(all_chunks)
    add_to_sparse(all_chunks)

    return len(all_chunks)


# Return all indexed sources stored in Chroma metadata used by router.py for source-scoped routing.
def get_all_indexed_sources() -> list[str]:
    stats = get_chroma_stats()
    return stats.get("sources", [])