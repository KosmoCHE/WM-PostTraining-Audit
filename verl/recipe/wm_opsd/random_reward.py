"""I.i.d. Bernoulli reward used by the paper's COIN condition.

The reward is independent of the response, target, and environment. Its support
matches the binary observation reward used by GT and MIS. Set
``RWML_RANDOM_REWARD_P`` (default 0.5) and ``RWML_RANDOM_REWARD_SEED`` in each
Ray reward worker's environment.

Worker PIDs are mixed into the base seed to prevent separate workers from
replaying identical random streams. The Bernoulli distribution is reproducible,
but concurrent calls are not guaranteed to be bitwise reproducible.
"""
import os
import random

_RNG = None
_P = None


def _get_rng() -> random.Random:
    """Create one independent random stream per reward-worker process."""
    global _RNG
    if _RNG is None:
        base = int(os.environ.get("RWML_RANDOM_REWARD_SEED", "0"))
        _RNG = random.Random(base * 1000003 + os.getpid())
    return _RNG


def _get_p() -> float:
    global _P
    if _P is None:
        _P = float(os.environ.get("RWML_RANDOM_REWARD_P", "0.5"))
        if not 0.0 <= _P <= 1.0:
            raise ValueError(f"RWML_RANDOM_REWARD_P must be in [0, 1], got {_P}")
    return _P


async def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float:
    """Return a Bernoulli reward without reading the response or target."""
    return 1.0 if _get_rng().random() < _get_p() else 0.0
