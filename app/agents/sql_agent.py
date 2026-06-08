import time
from langchain.tools import tool
from app.logging.logger import logger
from app.llm import llm

_SYSTEM = (
    "You are an expert database engineer. When converting natural language to SQL: "
    "assume a reasonable schema if none is provided, write ANSI-compatible SQL unless a dialect is specified, "
    "prefer CTEs over nested subqueries, add inline comments for non-obvious parts. "
    "When explaining SQL, go step by step. Always wrap SQL in ```sql ... ``` blocks."
)


def run_sql_agent(query: str, history: list[dict] | None = None) -> str:
    t0 = time.time()
    logger.info(f"SQL AGENT -> {query}")

    messages = [{"role": "system", "content": _SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    answer = llm.invoke(messages).content.strip()
    logger.info(f"SQL AGENT DONE -> {round(time.time()-t0, 2)}s")
    return answer


@tool
def sql_agent(query: str) -> str:
    """Write SQL, explain queries, or design schemas from natural language."""
    return run_sql_agent(query)