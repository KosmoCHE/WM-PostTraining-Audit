"""Within-group advantage permutation for the GT-PERM and MIS-PERM controls.

The estimator first computes ordinary group-normalized GRPO advantages and then
permutes those scalar advantages among responses that share a prompt UID. This
preserves the reward and advantage multisets exactly while breaking the
response-level reward assignment.

Register the estimator by setting
``VERL_USE_EXTERNAL_MODULES=recipe.wm_opsd.rwml_perm_adv`` and select it with
``algorithm.adv_estimator=grpo_perm``. ``RWML_PERM_SEED`` controls the fixed
base seed. A module-level call counter derives a different deterministic
permutation for each training step; resuming in the middle of a run resets that
counter.
"""
import hashlib
import os
from collections import defaultdict

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est

# The single driver process calls this estimator once per training step.
_CALL_COUNT = 0
_SEED_LOGGED = False


def _resolve_base_seed() -> int:
    return int(os.environ.get("RWML_PERM_SEED", "0"))


def _adv_hash(scalars: torch.Tensor) -> str:
    """Hash sorted scalar advantages to verify exact multiset preservation."""
    vals = torch.sort(scalars).values.detach().cpu().numpy().astype(np.float64)
    return hashlib.md5(vals.tobytes()).hexdigest()[:12]


@register_adv_est("grpo_perm")
def compute_grpo_perm_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
):
    """Compute GRPO advantages, then permute them within each prompt UID."""
    global _CALL_COUNT, _SEED_LOGGED

    if index is None:
        raise ValueError("grpo_perm requires prompt UID indices")

    # External estimators read this setting from the config when available.
    norm = norm_adv_by_std_in_grpo
    if config is not None and hasattr(config, "get"):
        norm = config.get("norm_adv_by_std_in_grpo", norm_adv_by_std_in_grpo)

    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    bsz = scores.shape[0]

    with torch.no_grad():
        # Match the standard GRPO group normalization.
        id2score = defaultdict(list)
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        id2mean, id2std = {}, {}
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                s = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(s)
                id2std[idx] = torch.std(s)
        adv_orig = scores.clone()
        for i in range(bsz):
            if norm:
                adv_orig[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                adv_orig[i] = scores[i] - id2mean[index[i]]

        # Preserve first-seen group order and permute only within each group.
        base_seed = _resolve_base_seed()
        if not _SEED_LOGGED:
            print(f"[rwml_perm] base_seed={base_seed} (RWML_PERM_SEED)", flush=True)
            _SEED_LOGGED = True
        rng = np.random.default_rng([base_seed, _CALL_COUNT])
        step = _CALL_COUNT
        _CALL_COUNT += 1

        groups = defaultdict(list)  # uid -> batch positions, in first-seen order
        for i in range(bsz):
            groups[index[i]].append(i)

        adv_perm = adv_orig.clone()
        moved = 0
        for pos in groups.values():
            if len(pos) <= 1:
                continue
            perm = rng.permutation(len(pos))
            src = torch.tensor([pos[p] for p in perm], device=adv_orig.device)
            dst = torch.tensor(pos, device=adv_orig.device)
            adv_perm[dst] = adv_orig[src]
            moved += int((perm != np.arange(len(pos))).sum())

        # Fail loudly if permutation changes values or introduces non-finite data.
        h_before, h_after = _adv_hash(adv_orig), _adv_hash(adv_perm)
        if h_before != h_after:
            raise RuntimeError(
                f"[rwml_perm audit] step={step} changed the global advantage multiset "
                f"({h_before} != {h_after})"
            )
        if not torch.isfinite(adv_perm).all():
            raise RuntimeError(f"[rwml_perm audit] step={step} produced NaN or Inf advantages")

        n_groups = len(groups)
        sizes = [len(p) for p in groups.values()]
        pos_rate = float((scores > 0).float().mean())
        print(
            f"[rwml_perm audit] step={step} seed={base_seed} n_groups={n_groups} "
            f"gsize[min/max]={min(sizes)}/{max(sizes)} pos_rate={pos_rate:.4f} "
            f"hash_ok={h_before == h_after} moved_frac={moved / bsz:.4f}",
            flush=True,
        )

        advantages = adv_perm.unsqueeze(-1) * response_mask

    return advantages, advantages
