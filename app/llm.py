from langchain_ollama import ChatOllama
from app.config import LLM_MODEL

# keep_alive=-1 keeps the model loaded indefinitely and prevents cold-start
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    num_predict=600,
    keep_alive=-1,
)

# same model, lower token budget - used for rewrite, file QA, critic
fast_llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    num_predict=400,
    keep_alive=-1,
)
