"""
Warm up LLM and classifier anchor embeddings at startup.
Both block until ready so the first real query has no cold-start cost.
"""
import time
from app.logging.logger import logger


def warmup_models():
    from app.llm import llm, fast_llm

    for name, model in [("llm", llm), ("fast_llm", fast_llm)]:
        for attempt in range(3):
            try:
                result = model.invoke("hi")
                if result and result.content:
                    logger.info(f"WARMUP -> {name} ready")
                    break
                time.sleep(1)
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"WARMUP -> {name} attempt {attempt+1} failed: {e}, retrying...")
                    time.sleep(2)
                else:
                    logger.error(f"WARMUP -> {name} failed after 3 attempts: {e}")

    # Trigger anchor embedding pre-computation
    from app.agents.classifier import _ANCHOR_EMBEDDINGS
    logger.info(f"WARMUP -> classifier anchors loaded for labels: {list(_ANCHOR_EMBEDDINGS.keys())}")
