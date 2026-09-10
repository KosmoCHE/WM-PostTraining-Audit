# Input data formats

This release does not redistribute benchmark trajectories. The preprocessing commands expect the following structures.

## ALFWorld trajectory JSONL

Each line is one episode:

```json
{
  "task": "put a clean mug in the cabinet",
  "metadata": {
    "gamefile": "...",
    "task_type": "pick_clean_then_place_in_recep",
    "split": "train",
    "variant": "give",
    "seed_tag": "run-0",
    "won": true
  },
  "transitions": [
    {
      "step": 0,
      "observation": "You are in the kitchen.",
      "action": "go to cabinet 1",
      "next_observation": "The cabinet 1 is closed.",
      "reward": 0,
      "done": false
    }
  ]
}
```

Extra transition metadata is retained when available but is not required for training.

## ScienceWorld trajectory JSONL

Each line contains `metadata` and `transitions`. The first transition's `observation` must contain the task description on the first line and the initial room observation on the remaining lines:

```json
{
  "metadata": {
    "task_idx": 12,
    "item_id": "sciworld_12",
    "split": "train",
    "success": true,
    "rollout_index": 0
  },
  "transitions": [
    {
      "observation": "Your task is to ...\nYou are in the workshop.",
      "action": "look around",
      "next_observation": "This room is called the workshop.",
      "reward": "0",
      "done": "False"
    }
  ]
}
```

## Text intermediate JSONL

Both builders emit one transition per line:

```json
{
  "prompt": "...",
  "target_next_state": "...",
  "meta": {
    "split": "train",
    "step": 0
  }
}
```

The MIS builder replaces `target_next_state` and stores the original target in `meta.true_next_state`.

## VisualWebArena rollout directory

```text
<dump-root>/<run-id>/episodes/<episode-id>/
  traj.json
  obs_0.jpg
  obs_1.jpg
  ...
```

`traj.json` contains `meta`, `snapshots`, and `steps`. A step references before/after snapshot indices and contains `operation` plus `action`; each snapshot contains `url`, `som_text`, and `image_file`. The builder keeps parseable transitions with an after snapshot, filters invalid element IDs and repeated no-op transitions, and embeds the current screenshot bytes in the output parquet.
