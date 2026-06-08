import time
from langchain.tools import tool
from app.logging.logger import logger
from app.llm import fast_llm

_SYSTEM = (
    "You are a concise, knowledgeable assistant. "
    "Answer clearly using your own knowledge. Do not make up facts. Be direct."
)


def run_gk_agent(query: str, history: list[dict] | None = None) -> str:
    t0 = time.time()
    logger.info(f"GENERAL KNOWLEDGE AGENT -> {query}")

    messages = [{"role": "system", "content": _SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    try:
        answer = fast_llm.invoke(messages).content.strip()
    except Exception as e:
        logger.error(f"GENERAL KNOWLEDGE AGENT ERROR -> {e}")
        answer = "I encountered an error answering that question."

    logger.info(f"GENERAL KNOWLEDGE AGENT DONE -> {round(time.time()-t0, 2)}s")
    return answer


@tool
def general_knowledge_agent(query: str) -> str:
    """Answer factual, conceptual, or explanatory questions from parametric knowledge."""
    return run_gk_agent(query)
