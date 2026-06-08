from __future__ import annotations
import time
from typing import Tuple, Dict

from app.agents.classifier import classify_query
from app.agents.web_search_agent import run_web_search_agent
from app.agents.coding_agent import run_coding_agent
from app.agents.sql_agent import run_sql_agent
from app.agents.general_knowledge_agent import run_gk_agent
from app.agents.math_agent import run_math_agent
from app.indexing.indexer import get_all_indexed_sources
from app.query.rewrite import rewrite_query_with_context
from app.storage.redis_cache import get_cache, set_cache
from app.utils.filename import normalize_filename, source_matches_query
from app.logging.logger import logger

_RAG_EMPTY_SIGNALS = [
    "i could not find", "i couldn't find", "no relevant",
    "not enough reliable", "insufficient information", "no information",
    "cannot find", "don't have information", "do not contain enough information",
]

_FAILED_WEB_SIGNALS = [
    "could not find live information",
    "could not find this information",
]

_DOC_INTENT_PHRASES = [
    "this document", "the document", "this file", "the file",
    "this text", "the text", "this pdf", "the pdf",
    "this report", "the report", "this paper", "the paper",
    "this resume", "the resume", "my resume",
    "i uploaded", "i've uploaded", "i shared", "i provided",
    "uploaded file", "uploaded document", "my document", "my file",
    "summarize the", "summarise the", "summarize this", "summarise this",
    "what does the", "what does this", "what is the document",
    "tell me about the", "explain the document", "explain this document",
    "what's in the", "what is in the",
]

_FILE_CTX_CHAR_LIMIT = 6_000


def _is_doc_directed(query: str) -> bool:
    q = query.lower()
    return any(phrase in q for phrase in _DOC_INTENT_PHRASES)


def _web_failed(answer: str) -> bool:
    return any(s in answer.lower() for s in _FAILED_WEB_SIGNALS)


def _rag_is_empty(answer: str) -> bool:
    if not answer or len(answer.strip()) < 80:
        return any(s in answer.lower() for s in _RAG_EMPTY_SIGNALS)
    return False


def _file_context_answer(query: str, file_ctx: Dict[str, str]) -> str:
    from app.llm import fast_llm
    matched = {f: t for f, t in file_ctx.items() if source_matches_query(f, query)}
    target  = matched if matched else file_ctx
    sections = []
    for fname, text in target.items():
        snippet = text[:_FILE_CTX_CHAR_LIMIT]
        if len(text) > _FILE_CTX_CHAR_LIMIT:
            snippet += "\n[... content truncated ...]"
        sections.append(f"=== {fname} ===\n{snippet}")
    prompt = (
        "You are a document QA assistant. Answer using ONLY the document content below. "
        "Be concise. If the answer is not present, say so.\n\n"
        f"Documents:\n{chr(10).join(sections)}\n\nQuestion: {query}\n\nAnswer:"
    )
    return fast_llm.invoke(prompt).content.strip()


def _run_rag(query: str, rag_agent, session_messages, system_prompt, filter_source=None) -> tuple[str, int]:
    try:
        invoke_input = {"user_question": query, "retry_count": 0}
        if filter_source:
            invoke_input["filter_source"] = filter_source
        response   = rag_agent.invoke(invoke_input)
        rag_answer = response.get("answer", "")
        chunks     = response.get("retrieved_chunks", "")
        doc_count  = len([c for c in chunks.split("\n\n---\n\n") if c.strip()]) if chunks and not chunks.startswith("I couldn't find") else 0
        return rag_answer, doc_count
    except Exception as e:
        logger.error(f"RAG AGENT ERROR -> {e}")
        return "", 0


def _find_mentioned_source(query: str, indexed_sources: list[str]) -> str | None:
    for src in indexed_sources:
        if source_matches_query(src, query):
            return src
    return None


def route_and_run(
    query: str,
    rag_agent,
    session_messages: list,
    system_prompt: dict,
    file_ctx: Dict[str, str] = None,
) -> Tuple[str, str, int]:
    """Returns (answer, agent_label, retrieved_docs_count)."""
    t0        = time.time()
    file_ctx  = file_ctx or {}
    has_files = bool(file_ctx)

    logger.info(f"ROUTER -> query: {query!r} | uploaded files: {list(file_ctx.keys())}")

    cached = get_cache(query)
    if cached:
        logger.info("ROUTER -> CACHE HIT")
        return cached["result"], f"⚡ Cache ({cached.get('agent', '📚 RAG Agent')})", 0

    # Session-file intercept
    if has_files and _is_doc_directed(query):
        session_match = next((f for f in file_ctx if source_matches_query(f, query)), None)
        if session_match:
            logger.info(f"ROUTER -> SESSION FILE MATCH '{session_match}' -> FILE CONTEXT")
            try:
                answer = _file_context_answer(query, {session_match: file_ctx[session_match]})
                if answer and answer.strip():
                    return answer, "📄 File Context Agent", 0
            except Exception as e:
                logger.error(f"FILE CONTEXT ERROR -> {e}")
        else:
            indexed_sources  = get_all_indexed_sources()
            mentioned_db_src = _find_mentioned_source(query, indexed_sources)
            session_names    = {normalize_filename(f)[0] for f in file_ctx}
            names_non_session = mentioned_db_src and normalize_filename(mentioned_db_src)[0] not in session_names

            logger.info(f"ROUTER DEBUG -> mentioned_db_source={mentioned_db_src} | names_non_session={names_non_session}")

            if len(file_ctx) == 1 and not names_non_session:
                logger.info("ROUTER -> GENERIC DOC QUERY + ONE SESSION FILE -> FILE CONTEXT")
                try:
                    answer = _file_context_answer(query, file_ctx)
                    if answer and answer.strip():
                        return answer, "📄 File Context Agent", 0
                except Exception as e:
                    logger.error(f"FILE CONTEXT ERROR -> {e}")
            else:
                logger.info("ROUTER -> QUERY REFERENCES DB-ONLY FILE -> FALL THROUGH TO RAG")

    # Build conversation context for classify + rewrite
    conversation_context = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in session_messages[-6:]
    )

    # Conversation history in LLM message format - passed to agents for multi-turn memory.
    # Exclude system messages; keep last 6 turns (3 user + 3 assistant).
    agent_history = [
        {"role": m["role"], "content": m["content"]}
        for m in session_messages[-6:]
        if m.get("role") in ("user", "assistant")
    ]

    rewritten_query = rewrite_query_with_context(query,conversation_context)
    label = classify_query(rewritten_query, has_files=has_files, file_names=list(file_ctx.keys()), conversation_context=conversation_context)
    logger.info(f"ROUTER -> classified as '{label}'")

    # Math queries must use the original query only
    effective_query = query if label == "math" else rewritten_query

    # Dispatch - all agents receive effective_query
    if label == "sql":
        answer = run_sql_agent(effective_query, history=agent_history)
        if answer:
            set_cache(query, {"result": answer, "agent": "🗄️ SQL Agent"})
            return answer, "🗄️ SQL Agent", 0

    elif label == "coding":
        answer = run_coding_agent(effective_query, history=agent_history)
        if answer:
            set_cache(query, {"result": answer, "agent": "💻 Coding Agent"})
            return answer, "💻 Coding Agent", 0

    elif label == "web":
        answer = run_web_search_agent(effective_query, history=agent_history)
        if answer and not _web_failed(answer):
            set_cache(query, {"result": answer, "agent": "🌐 Web Search Agent"})
            return answer, "🌐 Web Search Agent", 0

    elif label == "knowledge":
        answer = run_gk_agent(effective_query, history=agent_history)
        if answer:
            set_cache(query, {"result": answer, "agent": "🧠 General Knowledge Agent"})
            return answer, "🧠 General Knowledge Agent", 0

    elif label == "math":
        answer = run_math_agent(effective_query, history=agent_history)
        if answer:
            set_cache(query, {"result": answer, "agent": "🔢 Math Agent"})
            return answer, "🔢 Math Agent", 0

    elif label == "file" and has_files:
        try:
            answer = _file_context_answer(effective_query, file_ctx)
            if answer and answer.strip():
                return answer, "📄 File Context Agent", 0
        except Exception as e:
            logger.error(f"FILE CONTEXT ERROR -> {e}")

    # RAG fallback
    logger.info("ROUTER -> RAG AGENT")
    indexed    = get_all_indexed_sources()
    filter_src = _find_mentioned_source(effective_query, indexed)
    if filter_src:
        logger.info(f"ROUTER -> RAG scoped to source: {filter_src}")

    rag_answer, rag_doc_count = _run_rag(effective_query, rag_agent, session_messages, system_prompt, filter_source=filter_src)
    logger.info(f"ROUTER -> RAG ANSWER: {rag_answer[:120]!r} | docs retrieved: {rag_doc_count}")

    if rag_answer and not _rag_is_empty(rag_answer):
        set_cache(query, {"result": rag_answer, "agent": "📚 RAG Agent"})
        logger.info(f"ROUTER DONE -> RAG -> {round(time.time()-t0, 2)}s")
        return rag_answer, "📚 RAG Agent", rag_doc_count

    if has_files:
        logger.info("ROUTER -> RAG EMPTY + FILES PRESENT -> FILE CONTEXT FALLBACK")
        try:
            answer = _file_context_answer(effective_query, file_ctx)
            if answer and answer.strip():
                return answer, "📄 File Context Agent", 0
        except Exception as e:
            logger.error(f"FILE CONTEXT ERROR -> {e}")

    logger.info("ROUTER -> ALL FAILED -> FALLBACK WEB SEARCH")
    answer = run_web_search_agent(effective_query)
    if answer and answer.strip():
        set_cache(query, {"result": answer, "agent": "🌐 Web Search Agent"})
        return answer, "🌐 Web Search Agent", 0

    return "I could not find enough reliable information to answer that question.", "❌ No Agent", 0