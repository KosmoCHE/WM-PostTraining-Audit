import random
import unittest

from data_pipeline.alfworld.build_wm_samples import iter_wm_samples
from data_pipeline.shared.build_errframe_dataset import _replace_target, _target, derange
from data_pipeline.shared.build_verl_dataset import to_verl_row
from data_pipeline.vwa.build_wm_samples import format_history


class DataPipelineTest(unittest.TestCase):
    def test_alfworld_trajectory_expands_to_transition_samples(self):
        trajectory = {
            "task": "put the mug in the cabinet",
            "metadata": {
                "gamefile": "game.json",
                "task_type": "pick_and_place",
                "split": "train",
                "variant": "give",
                "seed_tag": "seed-0",
                "won": True,
            },
            "transitions": [
                {
                    "step": 0,
                    "observation": "At the counter.",
                    "action": "take mug 1 from counter 1",
                    "next_observation": "You pick up the mug 1.",
                    "reward": 0,
                    "done": False,
                }
            ],
        }
        samples = list(iter_wm_samples(trajectory))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["target_next_state"], "You pick up the mug 1.")
        self.assertIn("take mug 1 from counter 1", samples[0]["prompt"])

    def test_derangement_preserves_multiset_and_has_no_fixed_points(self):
        targets = ["a", "b", "c", "d"]
        permuted = derange(targets, random.Random(42))
        self.assertEqual(sorted(permuted), sorted(targets))
        self.assertTrue(all(a != b for a, b in zip(targets, permuted)))

    def test_mis_replacement_and_verl_conversion_use_intermediate_schema(self):
        row = {"prompt": "predict", "target_next_state": "true", "meta": {"split": "train"}}
        _replace_target(row, "mismatched")
        self.assertEqual(_target(row), "mismatched")
        self.assertEqual(row["meta"]["true_next_state"], "true")

        converted = to_verl_row(row, 7, "wm_test", tokenizer=None)
        self.assertEqual(converted["extra_info"]["target_next_state"], "mismatched")
        self.assertEqual(converted["extra_info"]["index"], 7)

    def test_vwa_history_contains_operation_and_action(self):
        steps = [
            {"parse_ok": True, "action": "click [3]", "operation": "ignored"},
            {"parse_ok": False, "action": None},
        ]
        self.assertEqual(format_history(steps), "1. ignored -> click [3]\n2. <invalid output>")


if __name__ == "__main__":
    unittest.main()
