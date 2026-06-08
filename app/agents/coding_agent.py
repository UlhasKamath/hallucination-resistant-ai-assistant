import time
from langchain.tools import tool
from app.logging.logger import logger
from app.llm import llm

_SYSTEM = (
    "You are an expert software engineer. Write clean, well-commented, production-quality code. "
    "When explaining, be clear and concise. When debugging, identify the root cause first. "
    "Always wrap code in markdown fenced blocks with the language tag."
)


def run_coding_agent(query: str, history: list[dict] | None = None) -> str:
    t0 = time.time()
    logger.info(f"CODING AGENT -> {query}")

    messages = [{"role": "system", "content": _SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    answer = llm.invoke(messages).content.strip()
    logger.info(f"CODING AGENT DONE -> {round(time.time()-t0, 2)}s")
    return answer


@tool
def coding_agent(query: str) -> str:
    """Write, explain, debug, or review code. Supports all common languages."""
    return run_coding_agent(query)
