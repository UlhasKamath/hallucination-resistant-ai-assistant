from sentence_transformers import CrossEncoder
from app.config import RERANK_MODEL

reranker = CrossEncoder(RERANK_MODEL)


# Queries asking to enumerate/list everything need broader coverage,
# not top-k similarity as reranking would discard relevant chunks.
_ENUM_PHRASES = [
    "list all", "list the", "what are all", "all questions", "all topics",
    "every question", "questions listed", "questions in", "questions are",
    "show all", "give me all", "what questions", "which questions",
]


def rerank(query, chunks, top_k=5):
    if not chunks:
        return []

    # For enumeration queries, skip reranking and return more chunks
    # so the generator has the full document spread, not just the top-5.
    q_lower = query.lower()
    if any(phrase in q_lower for phrase in _ENUM_PHRASES):
        return chunks[:max(top_k, 8)]

    pairs = [[query, c["text"]] for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [c for c, _ in ranked[:top_k]]