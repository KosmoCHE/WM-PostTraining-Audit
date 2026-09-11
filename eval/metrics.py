#!/usr/bin/env python
"""Compute task-level pass@k and empirical coverage from evaluator JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def pass_at_k(n: int, c: int, k: int) -> float:
    """Return the unbiased probability of at least one success in ``k`` draws."""
    if not 0 <= c <= n:
        raise ValueError(f"success count must be in [0, {n}], got {c}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(rows: list[dict], ks: list[int]) -> dict:
    """Aggregate standardized evaluator rows by task ID."""
    by_task = defaultdict(list)
    skipped_errors = 0
    for row in rows:
        if row.get("done_reason") == "error":
            skipped_errors += 1
            continue
        by_task[row["task_id"]].append((int(row["sample"]), bool(row["success"])))

    runs_by_task = {
        task_id: [success for _, success in sorted(samples)]
        for task_id, samples in by_task.items()
    }

    result = {"n_tasks": len(runs_by_task), "skipped_errors": skipped_errors, "metrics": {}}
    for k in ks:
        eligible = [runs for runs in runs_by_task.values() if len(runs) >= k]
        if not eligible:
            raise ValueError(f"no task has at least {k} samples")
        estimates = [pass_at_k(len(runs), sum(runs), k) for runs in eligible]
        first_k_coverage = sum(any(runs[:k]) for runs in eligible)
        result["metrics"][str(k)] = {
            "pass_at_k": sum(estimates) / len(estimates),
            "coverage_tasks": first_k_coverage,
            "eligible_tasks": len(eligible),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="standardized evaluator JSONL")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--output", help="optional JSON summary path")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    summary = summarize(rows, args.k)
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        import os

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")


if __name__ == "__main__":
    main()
