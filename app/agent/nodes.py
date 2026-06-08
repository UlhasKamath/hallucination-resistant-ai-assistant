from app.agents.rag_agent import rag_search
from app.llm import fast_llm
from app.logging.logger import logger

_ENUM_PHRASES = [
    "list all", "list the", "all questions", "every question",
    "questions listed", "questions in", "which questions", "what questions",
    "show all", "give me all", "all topics", "everything in",
]


def _is_enum_query(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in _ENUM_PHRASES)


def retriever_node(state):
    query         = state["user_question"]
    filter_source = state.get("filter_source")

    if filter_source:
        logger.info(f"RETRIEVER NODE -> scoped to source: {filter_source}")

        # For enumeration queries, bypass similarity search entirely - fetch every chunk for this source so nothing is missed.
        if _is_enum_query(query):
            from app.retrieval.dense import fetch_all_chunks_for_source
            all_chunks = fetch_all_chunks_for_source(filter_source)
            logger.info(f"RETRIEVER NODE -> enum query, fetched all {len(all_chunks)} chunks")
            context = "\n\n---\n\n".join([r["text"] for r in all_chunks]) if all_chunks else "I couldn't find reliable information on that topic."
        else:
            from app.retrieval.pipeline import hybrid_pipeline
            results = hybrid_pipeline(query, filter_source=filter_source)
            context = "\n\n---\n\n".join([r["text"] for r in results]) if results else "I couldn't find reliable information on that topic."
    else:
        context = rag_search.invoke(query)

    logger.info(f"RETRIEVER NODE -> context length: {len(context)} chars")
    return {"retrieved_chunks": context}


def generator_node(state):
    query   = state["user_question"]
    context = state.get("retrieved_chunks", "").strip()

    if not context or context.startswith("I couldn't find"):
        return {"answer": "I could not find relevant information in the knowledge base for that question."}

    prompt = (
        "You are a document QA assistant. Answer using ONLY the retrieved context below.\n"
        "If the context lacks enough information, say: "
        "'The retrieved documents do not contain enough information to answer this question.'\n"
        "Do NOT use outside knowledge or make up information.\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    )
    answer = fast_llm.invoke(prompt).content.strip()
    logger.info(f"GENERATOR NODE -> answer length: {len(answer)} chars")
    return {"answer": answer}


def critic_node(state):
    answer  = state.get("answer", "")
    context = state.get("retrieved_chunks", "")
    retry   = state.get("retry_count", 0)

    if retry >= 1 or not context:
        return {"is_grounded": True, "retry_count": retry}

    _HONEST_REFUSALS = [
        "do not contain enough information", "could not find relevant information",
        "not enough information", "cannot answer", "no information",
    ]
    if any(r in answer.lower() for r in _HONEST_REFUSALS):
        return {"is_grounded": True, "retry_count": retry}

    prompt = (
        "Does this answer use only information from the context? Reply YES or NO only.\n\n"
        f"Context:\n{context[:1500]}\n\nAnswer:\n{answer}"
    )
    try:
        grounded = fast_llm.invoke(prompt).content.strip().lower().startswith("yes")
    except Exception:
        grounded = True

    if not grounded:
        logger.warning(f"CRITIC -> not grounded, retry {retry + 1}")

    return {"is_grounded": grounded, "retry_count": retry + (0 if grounded else 1)}