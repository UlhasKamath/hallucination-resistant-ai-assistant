import asyncio
from app.ingestion.search import search_web
from app.storage.document_store import store_document, seen_url


async def _scrape_async(urls):
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, url) for url in urls]
    results = await asyncio.gather(*tasks)

    return [(u, r) for u, r in zip(urls, results) if r]


async def ingest(query: str):
    urls = search_web(query)

    # Dedup URLs
    urls = [u for u in urls if not seen_url(u)]

    if not urls:
        return []

    scraped = await _scrape_async(urls)

    doc_ids = []
    for url, texts in scraped:
        if not texts:
            continue

        doc_id = store_document(url, texts)
        if doc_id:
            doc_ids.append(doc_id)

    return doc_ids