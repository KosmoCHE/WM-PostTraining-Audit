"""Share duplicate multimodal-input objects within each prompt UID group.

Multiple rollouts of the same prompt can contain byte-identical image tensors
stored as distinct Python objects. Replacing them with a shared reference lets
pickle memoization avoid repeated transport without changing worker inputs.

The optimization is disabled by default. Enable it with
``WM_OPSD_MM_REF_DEDUP=1``. Group members are collapsed only when their keys and
tensor shapes match. ``WM_OPSD_MM_BF16=1`` optionally casts transported pixel
values to bfloat16 before sharing them.
"""

import os

import torch

__all__ = ["maybe_collapse_mm_refs"]


def maybe_collapse_mm_refs(batch) -> dict:
    """Collapse duplicate multimodal inputs in place and return counters."""
    stats = {"groups": 0, "collapsed": 0, "skipped_shape": 0}
    if os.environ.get("WM_OPSD_MM_REF_DEDUP", "0") != "1":
        return stats
    ntb = batch.non_tensor_batch
    if "multi_modal_inputs" not in ntb or "uid" not in ntb:
        return stats
    mm = ntb["multi_modal_inputs"]
    uids = ntb["uid"]
    cast_bf16 = os.environ.get("WM_OPSD_MM_BF16", "0") == "1"

    first_by_uid: dict = {}
    for i in range(len(uids)):
        uid = uids[i]
        cur = mm[i]
        if uid not in first_by_uid:
            if (
                cast_bf16
                and isinstance(cur, dict)
                and torch.is_tensor(cur.get("pixel_values"))
                and cur["pixel_values"].dtype == torch.float32
            ):
                cur = dict(cur)
                cur["pixel_values"] = cur["pixel_values"].to(torch.bfloat16)
                mm[i] = cur
            first_by_uid[uid] = cur
            stats["groups"] += 1
            continue
        ref = first_by_uid[uid]
        if _same_structure(cur, ref):
            mm[i] = ref
            stats["collapsed"] += 1
        else:
            stats["skipped_shape"] += 1
    return stats


def _same_structure(cur, ref) -> bool:
    if not (isinstance(cur, dict) and isinstance(ref, dict)):
        return False
    if cur.keys() != ref.keys():
        return False
    for k in cur.keys():
        a, b = cur[k], ref[k]
        if torch.is_tensor(a) != torch.is_tensor(b):
            return False
        if torch.is_tensor(a):
            # A bfloat16 reference may replace a float32 object with the same shape.
            if a.shape != b.shape:
                return False
        elif a is not b and a != b:
            return False
    return True
