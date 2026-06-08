"""
Hybrid Retrieval Pipeline
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.retrieval.dense import dense_search
from app.retrieval.sparse import sparse_search
from app.retrieval.reranker import rerank
from app.query.rewrite import rewrite_query

_EXPAND = os.getenv("EXPAND_QUERIES", "false").lower() == "true"


def rrf(results_list, k=60):
    scores = {}
    for results in results_list:
        for rank, item in enumerate(results):
            key = item["chunk_id"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    lookup = {}
    for results in results_list:
        for item in results:
            lookup[item["chunk_id"]] = item

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [lookup[i] for i in ranked_ids if i in lookup]


def _search_pair(query: str, filter_source=None):
    """Run dense + sparse for one query concurrently, return (dense, sparse)."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_dense  = ex.submit(dense_search,  query, filter_source=filter_source)
        f_sparse = ex.submit(sparse_search, query, filter_source=filter_source)
        return f_dense.result(), f_sparse.result()


def hybrid_pipeline(query, filter_source=None):
    rq = rewrite_query(query)

    if _EXPAND:
        expanded   = expand_query(rq)
        all_queries = [rq] + expanded

        all_dense, all_sparse = [], []
        with ThreadPoolExecutor(max_workers=len(all_queries)) as ex:
            futures = {
                ex.submit(_search_pair, q, filter_source): q
                for q in all_queries
            }
            for fut in as_completed(futures):
                d, s = fut.result()
                all_dense.extend(d)
                all_sparse.extend(s)
    else:
        # Default fast path: single rewritten query, parallel dense+sparse
        all_dense, all_sparse = _search_pair(rq, filter_source)

    # Deduplicate
    seen = set()
    dense_dedup = [c for c in all_dense  if not (c["chunk_id"] in seen or seen.add(c["chunk_id"]))]
    seen = set()
    sparse_dedup = [c for c in all_sparse if not (c["chunk_id"] in seen or seen.add(c["chunk_id"]))]

    # RRF fusion → rerank
    fused    = rrf([dense_dedup, sparse_dedup])
    reranked = rerank(query, fused)
    return reranked
