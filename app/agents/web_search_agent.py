import time
from datetime import date
from langchain.tools import tool
from app.ingestion.search import search_web
from app.logging.logger import logger
from app.llm import fast_llm


_CONTENT_LIMIT = 300  # chars per result is enough for facts, avoids bloating prompt

def _fetch_web_context(query: str, max_results: int = 5) -> str:
    results = search_web(query, max_results=max_results)
    if not results:
        return ""
    parts = []
    for r in results:
        date_s  = r.get("published_date", "") or r.get("date", "")
        header  = f"[{date_s}] " if date_s else ""
        snippet = r.get("content", "")[:_CONTENT_LIMIT]
        parts.append(
            f"Title: {header}{r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content: {snippet}"
        )
    return "\n\n---\n\n".join(parts)


def _synthesise(query: str, context: str, history: list[dict] | None = None) -> str:
    if not context:
        return "I could not find live information on that topic right now."

    today = date.today().strftime("%B %d, %Y")
    system = (
        f"You are a web research assistant. Today is {today}.\n"
        "Answer using ONLY the search results provided. "
        "Do NOT add, infer, or invent information in the results. "
        "If results are partial, state what was found and suggest an official source. "
        "Prefer the most recently dated result if sources conflict."
    )

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": f"Question: {query}\n\nWeb Results:\n{context}\n\nAnswer:"})

    return fast_llm.invoke(messages).content.strip()


def run_web_search_agent(query: str, history: list[dict] | None = None) -> str:
    t0 = time.time()
    logger.info(f"WEB SEARCH AGENT -> {query}")
    try:
        answer = _synthesise(query, _fetch_web_context(query), history=history)
    except Exception as e:
        logger.error(f"WEB SEARCH AGENT ERROR -> {e}")
        answer = "I could not find live information on that topic right now."
    logger.info(f"WEB SEARCH AGENT DONE -> {round(time.time()-t0, 2)}s | answer: {answer[:80]!r}")
    return answer


@tool
def web_search_agent(query: str) -> str:
    """Search the live web. Use for current events, news, scores, prices, or anything time-sensitive."""
    return run_web_search_agent(query)