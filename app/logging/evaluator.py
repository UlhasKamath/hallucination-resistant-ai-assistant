import time
from app.logging.logger import logger

# Agents that don't retrieve docs will log N/A instead of 0 to avoid confusion
_NO_RETRIEVAL_AGENTS = {
    "💻 Coding Agent",
    "🗄️ SQL Agent",
    "🧠 General Knowledge Agent",
    "🌐 Web Search Agent",
    "📄 File Context Agent",
    "❌ No Agent",
}


def evaluate_response(
    query,
    response,
    start_time,
    retrieved_docs=None,
    agent_label: str = "",
):
    latency = round(time.time() - start_time, 2)

    if retrieved_docs is not None and retrieved_docs > 0:
        doc_count = retrieved_docs
    elif agent_label in _NO_RETRIEVAL_AGENTS:
        doc_count = "N/A"
    else:
        # Cache hit or RAG with 0 results then show 0
        doc_count = 0

    evaluation = {
        "query":            query,
        "agent":            agent_label or "unknown",
        "retrieved_docs":   doc_count,
        "response_length":  len(response),
        "latency_seconds":  latency,
    }

    logger.info(f"EVALUATION -> {evaluation}")
    return evaluation
