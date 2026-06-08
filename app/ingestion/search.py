from tavily import TavilyClient
from app.logging.logger import logger
from app.config import TAVILY_API_KEY

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY missing from environment.")

client = TavilyClient(api_key=TAVILY_API_KEY)


def search_web(query: str, max_results: int = 5) -> list:
    try:
        response = client.search(
            query=query,
            search_depth="advanced",   # "advanced" adds ~15s with minimal quality gain
            max_results=max_results
            # topic="news",
            # days=30
        )
        results = response.get("results", [])
        logger.info(f"SEARCH -> {len(results)} Tavily results")
        return results
    except Exception as e:
        logger.error(f"TAVILY SEARCH ERROR -> {e}")
        return []
