import time
import re
import numexpr
from langchain.tools import tool
from app.logging.logger import logger
from app.llm import fast_llm

_SYSTEM = (
    "You are a precise mathematics assistant. "
    "Solve the problem step by step, showing your working clearly. "
    "Always state the final answer explicitly at the end. "
    "Do not guess — if a problem is ambiguous, say so. "
    "Write all output as plain text. Do NOT use LaTeX notation such as \\boxed{}, "
    "\\frac{}, \\[ \\], or any other LaTeX commands."
)

_DIRECT_EXPR_RE = re.compile(r"^[\d\s\+\-\*\/\.\(\)\^%]+$")


def _try_numexpr(query: str) -> str | None:
    """Try to evaluate a bare numeric expression directly. Returns None if not applicable."""
    expr = query.strip().rstrip("=?").strip()

    # Replace ^ with ** for exponentiation
    expr = expr.replace("^", "**")

    if _DIRECT_EXPR_RE.match(expr):
        try:
            result = numexpr.evaluate(expr).item()
            result_str = str(int(result)) if isinstance(result, float) and result.is_integer() else str(result)
            logger.info(f"MATH AGENT -> numexpr direct eval: {expr} = {result_str}")
            return f"{expr.replace('**', '^')} = **{result_str}**"
        except Exception:
            return None
    return None


def run_math_agent(query: str, history: list[dict] | None = None) -> str:
    t0 = time.time()
    logger.info(f"MATH AGENT -> {query}")

    direct = _try_numexpr(query)
    if direct:
        logger.info(f"MATH AGENT DONE (direct) -> {round(time.time()-t0, 3)}s")
        return direct

    messages = [{"role": "system", "content": _SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    try:
        answer = fast_llm.invoke(messages).content.strip()
    except Exception as e:
        logger.error(f"MATH AGENT ERROR -> {e}")
        answer = "I encountered an error solving that problem."

    logger.info(f"MATH AGENT DONE -> {round(time.time()-t0, 2)}s")
    return answer


@tool
def math_agent(query: str) -> str:
    """Solve mathematical problems: arithmetic, algebra, calculus, statistics, word problems."""
    return run_math_agent(query)