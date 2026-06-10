from datetime import date
from app.llm import fast_llm
from app.logging.logger import logger


def rewrite_query(query: str) -> str:
    """Standalone rewrite for RAG graph nodes — no conversation context."""
    prompt = f"""
        You are a query rewriter.

        Rules:
        - Preserve the user's intent exactly.
        - Preserve all named entities exactly.
        - Do not introduce new topics, people, events, products, teams, or facts.
        - Do not answer the query. Never broaden the query.
        - If the query is already clear, return it unchanged.
        - Return only the rewritten query.

        Query: {query}
        """
    try:
        result = fast_llm.invoke(prompt).content.strip()
        if len(result) > len(query) * 3:
            return query
        return result if result else query
    except Exception:
        return query


def rewrite_query_with_context(query: str, conversation_context: str) -> str:
    """
    Resolve pronouns and implicit references in a follow-up query using
    recent conversation history. Returns the original query unchanged if
    no context is available or the rewrite fails.
    """
    if not conversation_context.strip():
        return query

    

    today = date.today().strftime("%B %d, %Y")
    prompt = (
        f"Today is {today}. Rewrite the follow-up into a fully self-contained, "
        "web-searchable question using the conversation history.\n"
        "Rules: return ONLY the rewritten question. If already self-contained, "
        "return it unchanged. Do not add facts not in the conversation. "
        "Never invent a year unless the conversation explicitly states one.\n\n"
        "Examples:\n"
        "Conversation: USER: Has France announced their 2026 World Cup squad? ASSISTANT: Yes.\n"
        "Follow-up: Name all the players\n"
        "Rewritten: Name all players in France's 2026 FIFA World Cup squad\n\n"
        "Conversation: USER: Explain Python decorators ASSISTANT: A decorator wraps a function.\n"
        "Follow-up: Show a real example\n"
        "Rewritten: Show a real Python decorator example\n\n"
        f"Conversation:\n{conversation_context}\n\n"
        f"Follow-up: {query}\n"
        "Rewritten:"
    )
    try:
        result = fast_llm.invoke(prompt).content.strip()
        if result and len(result) < 300:
            logger.info(f"QUERY REWRITE -> '{query}' => '{result}'")
            return result
    except Exception as e:
        logger.error(f"QUERY REWRITE ERROR -> {e}")
    return query
