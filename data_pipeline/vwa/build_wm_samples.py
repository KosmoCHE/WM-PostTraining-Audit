#!/usr/bin/env python
"""Build VisualWebArena COIN training parquet from trajectory dumps.

Each parseable step with an after snapshot becomes one transition sample. The
prompt contains the objective, operation/action history, current screenshot,
URL, and executed action. It deliberately contains no realized next observation.

The default cleaning procedure removes invalid element IDs, duplicate no-op
transitions, and duplicate ``(url_before, action)`` pairs. Drop counts are
written to a manifest beside the parquet file.

Example:
  python -m data_pipeline.vwa.build_wm_samples \
      --dump-root trajectories/vwa \
      --config-root visualwebarena/config_files/vwa \
      --runs collect_B_r0 collect_B_r1 collect_B_r2 \
      --out data/processed/vwa/coin.parquet \
      [--limit 100]
"""
import argparse
import glob
import hashlib
import json
import os

WM_PROMPT_TEMPLATE = """You are a world model for a web browsing environment. Given the objective, the action history, the current page screenshot (with numbered element marks), and the action just taken, predict the resulting next page.

Objective: {objective}

Action history:
{history}

Current page: <image>
URL: {url}

Action taken: {action}

First reason step-by-step about the effect of the action within <think> </think> tags. Then describe the predicted next page within <next_state> </next_state> tags."""


def format_history(steps_before: list[dict]) -> str:
    """Format operation/action history and mark unparseable model outputs."""
    if not steps_before:
        return "None (this is the first step)."
    lines = []
    for i, s in enumerate(steps_before, 1):
        if not s.get("parse_ok"):
            lines.append(f"{i}. <invalid output>")
            continue
        operation = (s.get("operation") or "").strip().replace("\n", " ")
        action = (s.get("action") or "").strip()
        lines.append(f"{i}. {operation} -> {action}" if operation else f"{i}. {action}")
    return "\n".join(lines)


_OBJECTIVE_CACHE: dict[tuple, str] = {}


def get_objective(site: str, task_id, config_root: str) -> str:
    """Load the task objective from generated VisualWebArena task configs."""
    key = (site, task_id)
    if key not in _OBJECTIVE_CACHE:
        cfg = os.path.join(config_root, f"test_{site}", f"{task_id}.json")
        with open(cfg, encoding="utf-8") as f:
            _OBJECTIVE_CACHE[key] = json.load(f)["intent"]
    return _OBJECTIVE_CACHE[key]


def iter_transitions(traj: dict, ep_dir: str, run_id: str, config_root: str):
    """Yield usable transitions from one episode."""
    snaps = traj["snapshots"]
    meta = traj["meta"]
    objective = get_objective(meta["site"], meta["task_id"], config_root)
    for s in traj["steps"]:
        if s.get("snapshot_after") is None or not s.get("parse_ok"):
            continue
        before = snaps[s["snapshot_before"]]
        after = snaps[s["snapshot_after"]]
        img_path = os.path.join(ep_dir, before["image_file"] or "")
        if not before["image_file"] or not os.path.exists(img_path):
            continue
        yield {
            "objective": objective,
            "history": format_history(traj["steps"][: s["idx"]]),
            "url_before": before["url"],
            "url_after": after["url"],
            "action": s["action"],
            "invalid_id": s.get("invalid_id"),
            "som_before_hash": hashlib.md5(
                (before["som_text"] or "").encode()).hexdigest(),
            "som_after_hash": hashlib.md5(
                (after["som_text"] or "").encode()).hexdigest(),
            "image_path": img_path,
            "loc": {"run_id": run_id,
                    "episode": os.path.basename(ep_dir),
                    "step_idx": s["idx"],
                    "snapshot_after": s["snapshot_after"]},
            "site": meta.get("site"),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-root", required=True)
    ap.add_argument(
        "--config-root",
        required=True,
        help="VisualWebArena per-task config root containing test_<site>/<task_id>.json",
    )
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-clean", action="store_true",
                    help="disable invalid-action and duplicate-transition filtering")
    args = ap.parse_args()

    stats = {"episodes": 0, "raw": 0, "drop_invalid_id": 0,
             "drop_noop_dup": 0, "drop_url_action_dup": 0, "kept": 0}
    noop_seen, ua_seen = set(), set()
    samples = []

    for run_id in args.runs:
        for tj in sorted(glob.glob(
                os.path.join(args.dump_root, run_id, "episodes", "*", "traj.json"))):
            ep_dir = os.path.dirname(tj)
            try:
                with open(tj, encoding="utf-8") as f:
                    traj = json.load(f)
            except json.JSONDecodeError:
                continue
            stats["episodes"] += 1
            for tr in iter_transitions(traj, ep_dir, run_id, args.config_root):
                stats["raw"] += 1
                if not args.no_clean:
                    if tr["invalid_id"] is True:
                        stats["drop_invalid_id"] += 1
                        continue
                    # Keep only the first no-op for each URL/action pair.
                    if tr["som_before_hash"] == tr["som_after_hash"] \
                            and tr["url_before"] == tr["url_after"]:
                        key = (tr["url_before"], tr["action"])
                        if key in noop_seen:
                            stats["drop_noop_dup"] += 1
                            continue
                        noop_seen.add(key)
                    # Deduplicate the remaining transitions globally by URL/action.
                    ua = (tr["url_before"], tr["action"])
                    if ua in ua_seen:
                        stats["drop_url_action_dup"] += 1
                        continue
                    ua_seen.add(ua)
                stats["kept"] += 1
                samples.append(tr)
                if args.limit and len(samples) >= args.limit:
                    break
            if args.limit and len(samples) >= args.limit:
                break
        if args.limit and len(samples) >= args.limit:
            break

    print(f"stats: {json.dumps(stats, ensure_ascii=False)}")
    if stats["raw"]:
        print(f"clean ratio: kept {stats['kept']}/{stats['raw']} "
              f"({100*stats['kept']/stats['raw']:.1f}%)")

    rows = []
    for tr in samples:
        prompt_text = WM_PROMPT_TEMPLATE.format(
            objective=tr["objective"], history=tr["history"],
            url=tr["url_before"], action=tr["action"])
        with open(tr["image_path"], "rb") as f:
            img_bytes = f.read()
        rows.append({
            "data_source": "vwa_wm_coin",
            "prompt": [{"role": "user", "content": prompt_text}],
            "images": [{"bytes": img_bytes}],
            "ability": "world_model",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "site": tr["site"],
                "url_before": tr["url_before"],
                "url_after": tr["url_after"],
                "action": tr["action"],
                **tr["loc"],
            },
        })

    import pandas as pd
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {len(rows)} samples -> {args.out} ({size_mb:.1f} MB)")

    manifest = {"stats": stats, "runs": args.runs, "n_samples": len(rows),
                "cleaned": not args.no_clean, "limit": args.limit}
    with open(args.out + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
