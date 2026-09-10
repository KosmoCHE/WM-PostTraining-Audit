#!/usr/bin/env python
"""Build transition-level world-model samples from ALFWorld trajectories.

Every transition becomes one ``(history, observation, action) -> next state``
sample. Failed actions and unsuccessful trajectories are retained because they
still describe valid environment transitions. The complete observation/action
history is preserved without reasoning traces or a sliding window.

The output is a self-contained JSONL intermediate format. Convert it to verl
parquet with ``data_pipeline.shared.build_verl_dataset``.

Example:
  python -m data_pipeline.alfworld.build_wm_samples \
      --in trajectories/alfworld_train.jsonl \
      --out work/alfworld/wm_samples.jsonl
"""
import argparse
import json
import os


# RWML-style prediction prompt using history, current observation, and action.
WM_PROMPT_TEMPLATE = """\
You are a world model for the ALFRED Embodied Environment. Given the interaction history, the current observation, and the action just taken, predict the resulting next observation.

Task: {task}

Interaction history:
{history}

Current observation: {current_observation}
Action taken: {action}

First reason step-by-step about the effect of the action within <think> </think> tags. Then give the predicted next observation within <next_state> </next_state> tags.
"""


def build_history(transitions, upto):
    """Build the full observation/action history before transition ``upto``."""
    if upto == 0:
        return "N/A (this is the first step)."
    lines = []
    for i in range(upto):
        t = transitions[i]
        lines.append(f"Observation {i}: {t['observation']}")
        act = t["action"] if t["action"] is not None else "(no valid action)"
        lines.append(f"Action {i}: {act}")
    return "\n".join(lines)


def iter_wm_samples(traj):
    """Yield one world-model sample for each usable transition."""
    m = traj["metadata"]
    task = traj["task"]
    trs = traj["transitions"]
    for t_idx, t in enumerate(trs):
        # A missing next observation cannot provide a training target.
        if t.get("next_observation") is None:
            continue
        history = build_history(trs, t_idx)
        action = t["action"] if t["action"] is not None else ""
        prompt = WM_PROMPT_TEMPLATE.format(
            task=task, history=history,
            current_observation=t["observation"], action=action)
        yield {
            "prompt": prompt,
            "target_next_state": t["next_observation"],
            "meta": {
                "gamefile": m["gamefile"],
                "task_type": m["task_type"],
                "split": m["split"],
                "variant": m["variant"],
                "seed_tag": m["seed_tag"],
                "traj_won": m["won"],
                "step": t["step"],
                "action": t["action"],
                "action_in_admissible": t.get("action_in_admissible"),
                "parse_ok": t.get("parse_ok"),
                "extract_method": t.get("extract_method"),
                "reward": t.get("reward"),
                "done": t.get("done"),
            },
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="ALFWorld trajectory JSONL")
    ap.add_argument("--out", required=True, help="output world-model sample JSONL")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_traj = n_sample = 0
    with open(args.inp, encoding="utf-8") as fin, \
            open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            n_traj += 1
            for sample in iter_wm_samples(traj):
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_sample += 1

    print(f"[done] {n_traj} trajectories -> {n_sample} WM samples")
    print(f"       {args.out}")


if __name__ == "__main__":
    main()
