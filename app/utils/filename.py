"""
Filename normalization utility.
Single source of truth used everywhere filenames need to be matched
against query text: router, dense retrieval, sidebar UI.
"""
from pathlib import Path


def normalize_filename(name: str):
    cleaned = name.replace("file://", "")

    base = Path(cleaned).name

    stem = (
        Path(base).stem
        .replace("_", " ")
        .replace("-", " ")
        .lower()
    )

    words = [w for w in stem.split() if len(w) > 3]

    return stem, words



def source_matches_query(source: str, query: str) -> bool:
    stem, words = normalize_filename(source)

    q = query.lower()

    return (
        stem in q
        or any(w in q for w in words)
    )
