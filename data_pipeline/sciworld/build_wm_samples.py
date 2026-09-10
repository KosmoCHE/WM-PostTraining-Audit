#!/usr/bin/env python
"""Build transition-level world-model samples from ScienceWorld trajectories.

PatchWorld places the task description on the first line of the initial
observation. This builder separates that line from the room description so the
result matches the ALFWorld intermediate schema. Optional item-id and initial-
observation filters prevent evaluation variations from entering training.

Example:
  python -m data_pipeline.sciworld.build_wm_samples \
      --in  external/patchworld-trajectories/sciworld/sciworld_traj_train.jsonl \
      --out data/wm_rollouts/sciworld/wm_train.jsonl
"""
import argparse
import json
import os

from data_pipeline.sciworld.prompts import WM_PROMPT_TEMPLATE


def split_task_and_first_obs(first_observation):
    """Split the task line from the initial room observation."""
    task_desc, _, room = first_observation.partition("\n")
    return task_desc.strip(), room


def build_history(transitions, upto, task_desc):
    """Build the full observation/action history before transition ``upto``."""
    if upto == 0:
        return "N/A (this is the first step)."
    lines = []
    for i in range(upto):
        t = transitions[i]
        obs = t["observation"]
        if i == 0:
            _, obs = split_task_and_first_obs(obs)
        lines.append(f"Observation {i}: {obs}")
        act = t["action"] if t["action"] else "(no valid action)"
        lines.append(f"Action {i}: {act}")
    return "\n".join(lines)


def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1")


def iter_wm_samples(traj):
    """Yield one world-model sample for each usable transition."""
    m = traj["metadata"]
    trs = traj["transitions"]
    if not trs:
        return
    task_desc, _ = split_task_and_first_obs(trs[0]["observation"])

    for t_idx, t in enumerate(trs):
        if t.get("next_observation") is None:
            continue
        history = build_history(trs, t_idx, task_desc)
        # Remove the task line from the first current observation.
        cur_obs = t["observation"]
        if t_idx == 0:
            _, cur_obs = split_task_and_first_obs(cur_obs)
        action = t["action"] if t["action"] else ""
        prompt = WM_PROMPT_TEMPLATE.format(
            task=task_desc, history=history,
            current_observation=cur_obs, action=action)
        yield {
            "prompt": prompt,
            "target_next_state": t["next_observation"],
            "meta": {
                "task_idx": m.get("task_idx"),
                "item_id": m.get("item_id"),
                "split": m.get("split"),
                "variant": "nolist",  # ScienceWorld exposes no admissible-action list.
                "traj_success": m.get("success"),
                "rollout_index": m.get("rollout_index"),
                "step": t_idx,
                "action": t["action"],
                "reward": _to_float(t.get("reward")),
                "done": _to_bool(t.get("done")),
            },
        }


def _first_frame_fingerprint(traj):
    """Use the complete initial observation as a variation-level split key."""
    trs = traj.get("transitions")
    if not trs:
        return None
    return trs[0].get("observation")


def load_holdout_fingerprints(paths):
    """Load initial-observation fingerprints from held-out trajectory files."""
    fps = set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fp = _first_frame_fingerprint(json.loads(line))
                if fp is not None:
                    fps.add(fp)
    return fps


def load_exclude_item_ids(path):
    """Load integer indices from AgentGym ``sciworld_<index>`` item IDs."""
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            iid = json.loads(line).get("item_id", "")
            if iid.startswith("sciworld_"):
                ids.add(int(iid.split("_")[-1]))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True,
                    help="one or more PatchWorld trajectory JSONL files")
    ap.add_argument("--out", required=True, help="output world-model sample JSONL")
    ap.add_argument(
        "--holdout", nargs="*", default=[],
        help="held-out trajectory JSONL files whose initial observations must be excluded",
    )
    ap.add_argument(
        "--exclude-item-ids", default=None,
        help="AgentGym test JSONL; trajectories with matching sciworld_<index> IDs are excluded",
    )
    args = ap.parse_args()

    holdout_fps = load_holdout_fingerprints(args.holdout) if args.holdout else set()
    if args.holdout:
        print(f"[holdout] excluding {len(holdout_fps)} initial-observation fingerprints")
    exclude_ids = load_exclude_item_ids(args.exclude_item_ids) if args.exclude_item_ids else set()
    if exclude_ids:
        print(f"[exclude] excluding {len(exclude_ids)} AgentGym item IDs")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_traj = n_sample = n_skip_leak = n_skip_id = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for inp in args.inp:
            with open(inp, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    traj = json.loads(line)
                    # Exclude AgentGym test variations by item ID.
                    if exclude_ids:
                        iid = traj.get("metadata", {}).get("item_id", "")
                        if iid.startswith("sciworld_") and int(iid.split("_")[-1]) in exclude_ids:
                            n_skip_id += 1
                            continue
                    # Optionally exclude held-out variations by exact initial observation.
                    if holdout_fps and _first_frame_fingerprint(traj) in holdout_fps:
                        n_skip_leak += 1
                        continue
                    n_traj += 1
                    for sample in iter_wm_samples(traj):
                        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        n_sample += 1

    if exclude_ids:
        print(f"[de-leak by item_id] skipped {n_skip_id} AgentGym test trajectories")
    if holdout_fps:
        print(f"[de-leak by fingerprint] skipped {n_skip_leak} held-out trajectories")
    print(f"[done] {n_traj} trajectories -> {n_sample} WM samples")
    print(f"       {args.out}")


if __name__ == "__main__":
    main()
