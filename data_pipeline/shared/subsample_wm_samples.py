#!/usr/bin/env python
"""Select a reproducible random subset of intermediate world-model samples.

Sampling happens before MIS construction so GT and MIS contain the same source
rows and differ only in their assigned targets.

Example:
  python -m data_pipeline.shared.subsample_wm_samples \
      --in  data/wm_rollouts/sciworld/wm_train.jsonl \
      --out data/wm_rollouts/sciworld/wm_train_20k.jsonl \
      --n 20000 --seed 42
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="intermediate world-model JSONL")
    ap.add_argument("--out", required=True, help="sampled output JSONL")
    ap.add_argument("--n", type=int, default=20000, help="number of rows to select")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(args.inp, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(line)

    total = len(rows)
    if args.n >= total:
        print(f"[warn] --n={args.n} >= total rows {total}; writing every row")
        sampled = rows
    else:
        rng = random.Random(args.seed)
        idxs = sorted(rng.sample(range(total), args.n))  # Preserve source order.
        sampled = [rows[i] for i in idxs]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for line in sampled:
            fout.write(line + "\n")

    print(f"[done] {total} -> {len(sampled)} samples (seed={args.seed}) -> {args.out}")


if __name__ == "__main__":
    main()
