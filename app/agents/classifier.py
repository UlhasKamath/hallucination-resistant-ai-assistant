from __future__ import annotations
import re
import time
import numpy as np
from typing import Literal
from langchain_ollama import OllamaEmbeddings
from app.config import EMBED_MODEL
# from app.storage.redis_cache import get_cache, set_cache
from app.logging.logger import logger

AgentLabel = Literal["sql", "coding", "web", "knowledge", "rag", "file", "math"]


_embedder = OllamaEmbeddings(model=EMBED_MODEL)

# Pre-screening (zero latency - checked before any embed call)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")

_MATH_EXPR_RE = re.compile(r"^[\d\s\+\-\*\/\.\(\)\^%]+[=?]?\s*$")

_KB_PHRASES = [
    "knowledge base", "knowledge-base", "in the kb", "the kb",
    "indexed documents", "indexed files", "the documents", "the files",
    "across all", "all documents", "all files",
    "what's stored", "what is stored",
    "what have you indexed", "what topics are covered", "what is in the",
]

_CURRENT_ROLE_PHRASES = [
    "who is the manager", "who is the head coach", "who is the coach",
    "who is the ceo", "who is the president", "who is the prime minister",
    "who is the chancellor", "who is the captain", "who is the owner",
    "who is the chairman", "who runs", "who leads",
    "current manager", "current coach", "current ceo",
    "current president", "current prime minister", "current captain",
]

# Few diverse examples each. The query embeds once and is compared against all anchors
_ANCHORS: dict[str, list[str]] = {
    "web": [
        "What are the latest news about the French Open?",
        "Who won last night's Champions League match?",
        "What is the current Bitcoin price?",
        "Has the England squad been announced for the World Cup?",
        "What are the betting odds for Wimbledon?",
        "What happened in the US election today?",
        "Show me today's Premier League standings",
        "What are the latest updates on the Gaza ceasefire?",
        "What is the stock price of Apple right now?",
        "Who scored in last night's game?",
    ],
    "coding": [
        "Write a Python function to reverse a linked list",
        "Debug this JavaScript: for(let i=0, i<5, i++)",
        "How do I implement a binary search algorithm?",
        "Explain how async/await works in Python",
        "Write a React component that fetches and displays data",
        "What is the difference between a list and a tuple in Python?",
        "How do I connect to a PostgreSQL database using SQLAlchemy?",
        "Refactor this code to use a decorator pattern",
        "Write a bash script to rename all files in a directory",
        "How do I handle exceptions in a FastAPI endpoint?",
    ],
    "sql": [
        "Write a SQL query to get all orders placed in the last 7 days",
        "How do I design a star schema for a sales database?",
        "Explain what a LEFT JOIN does",
        "Write SQL to find the top 5 customers by revenue",
        "How do I create an index in PostgreSQL?",
        "What is the difference between HAVING and WHERE in SQL?",
        "Convert this natural language to SQL: show me all users who signed up this month",
        "How do I optimise a slow SQL query?",
        "Write a CTE to calculate rolling 7-day averages",
        "Design a schema for a multi-tenant SaaS application",
    ],
    "knowledge": [
        "What is the capital of France?",
        "Explain how photosynthesis works",
        "What is the difference between TCP and UDP?",
        "Who was the first person to walk on the moon?",
        "Explain the theory of relativity in simple terms",
        "What are the pros and cons of microservices architecture?",
        "What is the boiling point of water in Fahrenheit?",
        "Who wrote Pride and Prejudice?",
        "What does DNA stand for?",
        "How do I improve my sleep quality?"
    ],
    "rag": [
        "What does the knowledge base say about our refund policy?",
        "Search the indexed documents for anything about GDPR compliance",
        "Summarise everything in the knowledge base about onboarding",
        "What topics are covered in the uploaded documents?",
        "Find information about pricing in the stored files",
        "What have you indexed so far?",
        "Give me a summary of all documents related to data pipelines",
        "What does the knowledge base say about SLAs?",
        "Is there anything in the documents about security protocols?",
        "What files have been uploaded to the knowledge base?",
    ],
    "file": [
        "What does my uploaded PDF say about chapter 3?",
        "Summarise the resume I just uploaded",
        "What are the key findings in this report?",
        "Tell me about the document I shared",
        "What skills does the candidate in this CV have?",
        "Explain the methodology section of this paper",
        "What does this contract say about termination clauses?",
        "Give me a summary of this uploaded file",
        "What are the main points in the document I provided?",
        "Does this PDF mention anything about pricing?",
    ],
    "math": [
        "What is 15% of 2847?",
        "Solve for x: 3x + 7 = 22",
        "Calculate the compound interest on $5000 at 8% for 3 years",
        "What is the derivative of x^3 + 2x^2 - 5x?",
        "Find the area of a circle with radius 7",
        "What is the mean and standard deviation of [4, 8, 15, 16, 23, 42]?",
        "A train travels 120km in 1.5 hours. What is its speed?",
        "Integrate 2x^2 + 3x - 1 with respect to x",
        "What is the square root of 1764?",
        "How many permutations are there of 5 items taken 3 at a time?",
    ],
}

# Pre-compute anchor embeddings at import time
_ANCHOR_EMBEDDINGS: dict[str, np.ndarray] = {}

def _init_anchors() -> None:
    """Embed all anchor sentences and cache as numpy arrays."""
    t0 = time.time()
    for label, sentences in _ANCHORS.items():
        vecs = _embedder.embed_documents(sentences)
        _ANCHOR_EMBEDDINGS[label] = np.array(vecs, dtype=np.float32)
    logger.info(f"CLASSIFIER -> anchor embeddings ready ({round(time.time()-t0, 2)}s)")

# Run at import — happens once when the app starts (same time as model warmup)
_init_anchors()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _embed_classify(query: str) -> tuple[AgentLabel, dict[str, float]]:
    q_vec = np.array(_embedder.embed_query(query), dtype=np.float32)
    scores = {}
    for label, anchor_vecs in _ANCHOR_EMBEDDINGS.items():
        sims = [_cosine_sim(q_vec, a) for a in anchor_vecs]
        scores[label] = float(np.mean(sorted(sims, reverse=True)[:3]))
    best = max(scores, key=lambda l: scores[l])
    logger.info(f"CLASSIFIER -> embed scores: { {l: round(s, 3) for l, s in sorted(scores.items(), key=lambda x: -x[1])} }")
    return best, scores


# Cache helpers
# _CACHE_PREFIX = "clf:"

# def _clf_cache_get(key: str) -> AgentLabel | None:
#     result = get_cache(_CACHE_PREFIX + key)
#     return result["result"] if result else None

# def _clf_cache_set(key: str, label: AgentLabel) -> None:
#     set_cache(_CACHE_PREFIX + key, {"result": label})


# Public API
def classify_query(
    query: str,
    has_files: bool = False,
    file_names: list[str] = None,
    conversation_context: str = "",
) -> AgentLabel:
    t0 = time.time()

    # Pre-screens: no latency, best confidence cases
    if _CODE_BLOCK_RE.search(query):
        logger.info("CLASSIFIER -> PRESCREEN: code block → coding")
        return "coding"

    q_lower = query.lower()

    if _MATH_EXPR_RE.match(query.strip()):
        logger.info("CLASSIFIER -> PRESCREEN: bare numeric expression → math")
        return "math"

    if any(phrase in q_lower for phrase in _KB_PHRASES) and not has_files:
        logger.info("CLASSIFIER -> PRESCREEN: KB phrase → rag")
        return "rag"

    if any(phrase in q_lower for phrase in _CURRENT_ROLE_PHRASES):
        logger.info("CLASSIFIER -> PRESCREEN: current-role phrase → web")
        return "web"

    # Cache check
    # cache_key = query + ("|files" if has_files else "") + "|" + conversation_context[-200:]
    # cached = _clf_cache_get(cache_key)
    # if cached:
    #     logger.info(f"CLASSIFIER -> CACHE HIT: {cached}")
    #     return cached

    label, scores = _embed_classify(query)

    if label == "file" and not has_files:
        label = max((l for l in scores if l != "file"), key=lambda l: scores[l])

    logger.info(f"CLASSIFIER -> '{label}' ({round(time.time()-t0, 3)}s)")
    # _clf_cache_set(cache_key, label)
    return label
