"""Binary observation-matching reward used by the paper's GT and MIS runs.

The student emits ``<think>...</think><next_state>...</next_state>``. The reward
is one when the cosine distance between the predicted and target next-state
embeddings is below ``TAU_D`` and zero otherwise. Qwen3-Embedding-8B served by
an OpenAI-compatible endpoint was used in the reported experiments.

verl's naive reward manager awaits one sample at a time, so ``compute_score``
is asynchronous. Set ``RWML_EMB_URL`` in every Ray worker's environment.
"""
import math
import os

import aiohttp

NEXT_STATE_OPEN = "<next_state>"
NEXT_STATE_CLOSE = "</next_state>"

TAU_D = 0.2  # Threshold used in the reported experiments.
DEFAULT_EMB_URL = "http://127.0.0.1:13151/v1/embeddings"

_SESSION = None


def _get_session():
    """Reuse one HTTP session per reward-worker process."""
    global _SESSION
    if _SESSION is None:
        _SESSION = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
    return _SESSION


def _extract_next_state(response: str) -> str | None:
    """Extract a non-empty ``<next_state>`` span from a student response."""
    i = response.find(NEXT_STATE_OPEN)
    if i == -1:
        return None
    j = response.find(NEXT_STATE_CLOSE, i)
    if j == -1:
        return None
    text = response[i + len(NEXT_STATE_OPEN) : j].strip()
    return text or None


async def _embed(url: str, texts: list[str]) -> list[list[float]]:
    """Call an OpenAI-compatible embedding endpoint with bounded retries."""
    session = _get_session()
    payload = {"model": "embedding", "input": texts}
    last_exc = None
    for attempt in range(5):
        try:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return [d["embedding"] for d in data["data"]]
        except Exception as e:  # Retry transient endpoint failures.
            last_exc = e
            import asyncio

            await asyncio.sleep(min(2**attempt, 15))
    raise last_exc


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float:
    """Return the binary cosine-distance reward for one generated response."""
    pred = _extract_next_state(solution_str or "")
    if pred is None:
        return 0.0
    target = (extra_info or {}).get("target_next_state")
    if not target:
        return 0.0
    url = os.environ.get("RWML_EMB_URL", DEFAULT_EMB_URL)
    emb_pred, emb_target = await _embed(url, [pred, target])
    d = 1.0 - _cos(emb_pred, emb_target)
    return 1.0 if d < TAU_D else 0.0
