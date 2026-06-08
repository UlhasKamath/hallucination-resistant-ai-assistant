import asyncio
import time
from langchain.tools import tool

from app.ingestion.pipeline import ingest
from app.indexing.indexer import index_documents
from app.retrieval.pipeline import hybrid_pipeline
from app.logging.logger import logger


@tool
def rag_search(query: str) -> str:
    """
    Hybrid RAG search over the indexed knowledge base.
    Accepts a plain string query. filter_source is injected separately
    via the graph state in retriever_node — not passed through this tool.
    """
    logger.info(f"RAG SEARCH -> {query!r}")
    t0 = time.time()

    results = hybrid_pipeline(query)
    logger.info(f"INITIAL RETRIEVAL COUNT -> {len(results) if results else 0}")

    if not results:
        logger.info("NO RESULTS -> STARTING WEB INGESTION")
        doc_ids = asyncio.run(ingest(query))
        if doc_ids:
            n = index_documents(doc_ids)
            logger.info(f"INDEXED {n} CHUNKS")
            results = hybrid_pipeline(query)
            logger.info(f"POST-INGEST RETRIEVAL COUNT -> {len(results) if results else 0}")

    if not results:
        logger.warning("NO RELIABLE INFORMATION FOUND")
        return "I couldn't find reliable information on that topic."

    logger.info(f"RAG SEARCH DONE -> {len(results)} chunks | {round(time.time()-t0, 2)}s")
    return "\n\n---\n\n".join([r["text"] for r in results])
