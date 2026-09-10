#!/usr/bin/env python
"""Create the template-disjoint VisualWebArena split used by the paper.

VisualWebArena has no training split, and parameterized tasks from the same
intent template are near duplicates. The split unit is therefore
``(site, intent_template_id)``. Templates are greedily balanced within strata
defined by site, evaluation type, and overall difficulty. Tasks that require a
Wikipedia deployment are excluded.

Outputs under ``--out-dir``:

* ``template_split.json``: split metadata and template assignments.
* ``tasks_collect.jsonl``: tasks on the data-collection side.
* ``tasks_eval.jsonl``: tasks on the held-out side, including pool membership.

Example:
  python -m data_pipeline.vwa.split_templates \
      --config-dir visualwebarena/config_files/vwa \
      --out-dir data/splits/vwa
"""
import argparse
import collections
import json
import os
import random

SITES = ["shopping", "reddit", "classifieds"]

DIFFICULTY_FIX = {"hrad": "hard", "mediun": "medium"}


def norm_difficulty(v):
    v = (v or "").strip().lower()
    return DIFFICULTY_FIX.get(v, v)


def eval_type_of(task):
    """Return a stable representation of one or more evaluation types."""
    return "+".join(sorted(task["eval"]["eval_types"]))


def _site_config(config_dir, site):
    for name in (f"test_{site}.json", f"test_{site}.raw.json"):
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"missing test_{site}.json (or .raw.json) under {config_dir}")


def load_tasks(config_dir):
    """Load the three website configs and exclude Wikipedia-dependent tasks."""
    tasks, excluded = [], []
    for site in SITES:
        with open(_site_config(config_dir, site), encoding="utf-8") as f:
            for t in json.load(f):
                row = {
                    "site": site,
                    "task_id": t["task_id"],
                    "template_id": t["intent_template_id"],
                    "intent": t["intent"],
                    "eval_type": eval_type_of(t),
                    "overall_difficulty": norm_difficulty(t.get("overall_difficulty")),
                    "reasoning_difficulty": norm_difficulty(t.get("reasoning_difficulty")),
                    "visual_difficulty": norm_difficulty(t.get("visual_difficulty")),
                    "require_reset": bool(t.get("require_reset")),
                    "has_image": bool(t.get("image")),
                }
                if any("wiki" in s for s in t["sites"]):
                    excluded.append(row)
                else:
                    tasks.append(row)
    return tasks, excluded


def majority(values):
    return collections.Counter(values).most_common(1)[0][0]


def split_templates(tasks, seed):
    """Greedily balance template groups between collection and evaluation."""
    by_tpl = collections.defaultdict(list)
    for t in tasks:
        by_tpl[(t["site"], t["template_id"])].append(t)

    buckets = collections.defaultdict(list)
    for key, ts in by_tpl.items():
        bucket = (ts[0]["site"], majority([t["eval_type"] for t in ts]),
                  majority([t["overall_difficulty"] for t in ts]))
        buckets[bucket].append((key, len(ts)))

    rng = random.Random(seed)
    assign = {}
    for bucket in sorted(buckets):
        tpls = buckets[bucket]
        rng.shuffle(tpls)  # Randomize ties between equal-size templates.
        tpls.sort(key=lambda x: -x[1])
        n = {"collect": 0, "eval": 0}
        for key, size in tpls:
            side = min(("collect", "eval"), key=lambda s: (n[s], rng.random()))
            assign[key] = side
            n[side] += size
    return assign, by_tpl


def sample_eval_pool(eval_tasks, pool_size, seed):
    """Sample the evaluation pool proportionally by site and evaluation type."""
    eligible = [t for t in eval_tasks if not t["require_reset"]]
    if pool_size >= len(eligible):
        return {(t["site"], t["task_id"]) for t in eligible}
    strata = collections.defaultdict(list)
    for t in eligible:
        strata[(t["site"], t["eval_type"])].append(t)
    rng = random.Random(seed + 1)
    pool, remainders = set(), []
    for key in sorted(strata):
        ts = sorted(strata[key], key=lambda t: (t["site"], t["task_id"]))
        rng.shuffle(ts)
        quota = pool_size * len(ts) / len(eligible)
        k = int(quota)
        pool.update((t["site"], t["task_id"]) for t in ts[:k])
        if len(ts) > k:
            remainders.append((quota - k, ts[k]))  # Complete quotas by largest remainder.
    remainders.sort(key=lambda x: -x[0])
    for _, t in remainders[: pool_size - len(pool)]:
        pool.add((t["site"], t["task_id"]))
    return pool


def dist_table(tasks):
    d = {
        "n_tasks": len(tasks),
        "n_templates": len({(t["site"], t["template_id"]) for t in tasks}),
        "by_site": dict(collections.Counter(t["site"] for t in tasks)),
        "by_eval_type": dict(collections.Counter(t["eval_type"] for t in tasks)),
        "by_difficulty": dict(collections.Counter(t["overall_difficulty"] for t in tasks)),
        "with_image": sum(t["has_image"] for t in tasks),
        "require_reset": sum(t["require_reset"] for t in tasks),
    }
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool-size", type=int, default=200)
    ap.add_argument("--config-dir", required=True, help="VisualWebArena aggregated task configs")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    tasks, excluded = load_tasks(args.config_dir)
    assign, by_tpl = split_templates(tasks, args.seed)

    for t in tasks:
        t["side"] = assign[(t["site"], t["template_id"])]
    collect = [t for t in tasks if t["side"] == "collect"]
    evals = [t for t in tasks if t["side"] == "eval"]
    pool = sample_eval_pool(evals, args.pool_size, args.seed)
    for t in evals:
        t["in_eval_pool"] = (t["site"], t["task_id"]) in pool

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "date": "2026-08-01",
        "seed": args.seed,
        "unit": "(site, intent_template_id)",
        "excluded_wikipedia_tasks": [(t["site"], t["task_id"]) for t in excluded],
        "stratify_by": "(site, majority eval_type, majority overall_difficulty)",
        "eval_pool_size": len(pool),
        "eval_pool_note": "proportional sampling by site/eval_type after excluding require_reset",
        "collect": dist_table(collect),
        "eval": dist_table(evals),
        "templates": {
            f"{site}/{tid}": {"side": assign[(site, tid)], "n_tasks": len(by_tpl[(site, tid)])}
            for (site, tid) in sorted(assign)
        },
    }
    with open(os.path.join(args.out_dir, "template_split.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    for name, rows in [("tasks_collect.jsonl", collect), ("tasks_eval.jsonl", evals)]:
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            for t in sorted(rows, key=lambda t: (t["site"], t["task_id"])):
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"[excluded] {len(excluded)} Wikipedia-dependent tasks")
    for side, rows in [("collect", collect), ("eval", evals)]:
        d = dist_table(rows)
        print(f"[{side}] {d['n_templates']} templates / {d['n_tasks']} tasks | "
              f"sites {d['by_site']} | difficulty {d['by_difficulty']} | "
              f"with image {d['with_image']} | require_reset {d['require_reset']}")
    print(f"[eval pool] {len(pool)} tasks after excluding require_reset")
    print(f"[out] {args.out_dir}/template_split.json + tasks_collect.jsonl + tasks_eval.jsonl")


if __name__ == "__main__":
    main()
