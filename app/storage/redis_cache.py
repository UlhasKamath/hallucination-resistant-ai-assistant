import json
import redis
import os
import re

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

CACHE_TTL = 3600  # 1 hour


def get_cache(key: str):
    key = normalize_query(key)
    value = redis_client.get(f"cache:{key}")
    if value:
        return json.loads(value)
    return None


def set_cache(key: str, value):
    key = normalize_query(key)
    
    redis_client.setex(
        f"cache:{key}",
        CACHE_TTL,
        json.dumps(value)
    )


def delete_cache(key: str):
    redis_client.delete(f"cache:{key}")


def get_cache_stats():

    keys = redis_client.keys("cache:*")

    return {
        "cached_queries": len(keys),
        "keys": [k.replace("cache:", "") for k in keys]
    }

def normalize_query(query: str) -> str:

    query = query.lower()

    # remove punctuation
    query = re.sub(r"[^\w\s]", "", query)

    # remove extra spaces
    query = " ".join(query.split())

    return query