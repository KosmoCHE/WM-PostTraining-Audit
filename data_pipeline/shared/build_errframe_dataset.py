#!/usr/bin/env python
"""Construct the MIS control with a fixed target-observation derangement.

The transform preserves the exact target-observation multiset while ensuring
that no row keeps its original target text. The original target is stored in
``meta.true_next_state`` for auditing.

Example:
  python -m data_pipeline.shared.build_errframe_dataset \
      --in  work/alfworld/gt.jsonl \
      --out work/alfworld/mis.jsonl \
      --seed 42
"""
import argparse
import json
import os
import random


def derange(targets, rng):
    """Return a permutation whose text differs from the original at every index."""
    perm = list(targets)
    rng.shuffle(perm)
    # Remove text-level fixed points by swapping two mutually compatible rows.
    n = len(perm)
    for _ in range(100):
        bad = [i for i in range(n) if perm[i] == targets[i]]
        if not bad:
            return perm
        for i in bad:
            for _ in range(1000):
                j = rng.randrange(n)
                if perm[j] != targets[i] and perm[i] != targets[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                    break
    raise RuntimeError("derangement did not converge; target duplication may be too high")


def _target(row):
    if "target_next_state" in row:
        return row["target_next_state"]
    return row["metadata"]["target_next_state"]


def _replace_target(row, fake):
    if "target_next_state" in row:
        meta = row.setdefault("meta", {})
        meta["true_next_state"] = row["target_next_state"]
        row["target_next_state"] = fake
    else:
        metadata = row["metadata"]
        metadata["true_next_state"] = metadata["target_next_state"]
        metadata["target_next_state"] = fake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="intermediate world-model JSONL")
    ap.add_argument("--out", required=True, help="output MIS JSONL")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(args.inp, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    targets = [_target(r) for r in rows]
    fakes = derange(targets, random.Random(args.seed))

    n_collide = sum(1 for t, f in zip(targets, fakes) if t == f)
    assert n_collide == 0, f"{n_collide} rows retained the original target"
    assert sorted(fakes) == sorted(targets), "MIS targets must preserve the target multiset"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for row, fake in zip(rows, fakes):
            _replace_target(row, fake)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[done] {len(rows)} samples, 0 fixed target texts -> {args.out}")


if __name__ == "__main__":
    main()
