"""Minimal ALFWorld runtime used by the task evaluator."""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass


SPLIT_TO_MODE = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}
_TASK_RE = re.compile(r"Your task is to:\s*(.+?)\s*$", re.S)


def _base_config() -> dict:
    data_root = os.environ.get("ALFWORLD_DATA")
    if not data_root:
        raise RuntimeError("Set ALFWORLD_DATA to the directory containing json_2.1.1")
    return {
        "dataset": {
            "data_path": f"{data_root}/json_2.1.1/train",
            "eval_id_data_path": f"{data_root}/json_2.1.1/valid_seen",
            "eval_ood_data_path": f"{data_root}/json_2.1.1/valid_unseen",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "env": {
            "type": "AlfredTWEnv",
            "regen_game_files": False,
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "general": {"training_method": "dagger", "random_seed": 42, "use_cuda": False},
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
    }


def _single_environment_class():
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    class SingleAlfredTWEnv(AlfredTWEnv):
        """Avoid scanning the complete dataset when evaluating one game file."""

        def __init__(self, config, train_eval="train"):
            self.config = config
            self.train_eval = train_eval
            self.goal_desc_human_anns_prob = config["env"]["goal_desc_human_anns_prob"]
            self.random_seed = 42
            self.game_files = []
            self.num_games = 0

    return SingleAlfredTWEnv


def parse_reset_observation(raw: str) -> tuple[str, str]:
    """Separate the task description from ALFWorld's initial observation."""
    match = _TASK_RE.search(raw)
    task = match.group(1).strip() if match else ""
    body = _TASK_RE.sub("", raw) if task else raw
    segments = [segment for segment in body.split("\n\n") if segment.strip()]
    segments = [segment for segment in segments if not segment.strip().startswith("-= Welcome")]
    return task, "\n\n".join(segments).strip()


def task_type_of(gamefile: str) -> str:
    """Extract the ALFWorld task family from a game-file path."""
    try:
        return gamefile.split(os.sep)[-3].split("-")[0]
    except IndexError:
        return "unknown"


@dataclass
class StepResult:
    observation: str
    admissible_actions: list[str]
    done: bool
    won: bool


class ALFWorldEnvironment:
    """Evaluate one game with the native ``put X in/on Y`` action interface."""

    def __init__(self, gamefile: str):
        env_class = _single_environment_class()
        environment = env_class(_base_config(), train_eval="eval_in_distribution")
        environment.game_files = [gamefile]
        environment.num_games = 1
        self._environment = environment.init_env(batch_size=1)
        self.gamefile = gamefile
        self.task_type = task_type_of(gamefile)

    def reset(self) -> tuple[str, str, list[str]]:
        observations, info = self._environment.reset()
        task, observation = parse_reset_observation(observations[0])
        return task, observation, info["admissible_commands"][0]

    def step(self, action: str | None) -> StepResult:
        observations, _, dones, info = self._environment.step([action or ""])
        return StepResult(
            observation=observations[0],
            admissible_actions=info["admissible_commands"][0],
            done=bool(dones[0]),
            won=bool(info.get("won", [False])[0]),
        )


def list_split_games(split: str, limit: int | None, seed: int) -> list[str]:
    """List and reproducibly shuffle all game files in an ALFWorld split."""
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    if split not in SPLIT_TO_MODE:
        raise ValueError(f"unknown ALFWorld split: {split}")
    environment = AlfredTWEnv(_base_config(), train_eval=SPLIT_TO_MODE[split])
    games = list(environment.game_files)
    del environment
    random.Random(seed).shuffle(games)
    return games if limit is None else games[:limit]
