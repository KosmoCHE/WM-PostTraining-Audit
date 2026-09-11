import unittest

from eval.alfworld.prompts import parse_output as parse_alfworld
from eval.metrics import pass_at_k, summarize
from eval.sciworld.prompts import parse_output as parse_sciworld
from eval.vwa.prompts import format_history, parse_response, tier_b_observation


class EvaluationTest(unittest.TestCase):
    def test_pass_at_k(self):
        self.assertEqual(pass_at_k(4, 0, 2), 0.0)
        self.assertEqual(pass_at_k(4, 4, 2), 1.0)
        self.assertAlmostEqual(pass_at_k(4, 1, 2), 0.5)

    def test_metric_summary_skips_scienceworld_infrastructure_errors(self):
        rows = [
            {"task_id": "a", "sample": 0, "success": True},
            {"task_id": "a", "sample": 1, "success": False},
            {"task_id": "b", "sample": 0, "success": False},
            {"task_id": "b", "sample": 1, "success": False},
            {"task_id": "b", "sample": 2, "success": False, "done_reason": "error"},
        ]
        result = summarize(rows, [1, 2])
        self.assertEqual(result["skipped_errors"], 1)
        self.assertEqual(result["metrics"]["2"]["coverage_tasks"], 1)

    def test_alfworld_action_parser(self):
        reasoning, action = parse_alfworld("<think>plan</think><action>look</action>")
        self.assertEqual(action, "look")
        self.assertIn("plan", reasoning)
        _, fallback = parse_alfworld("Plan first\nAction: inventory")
        self.assertEqual(fallback, "inventory")

    def test_scienceworld_action_parser(self):
        reasoning, action = parse_sciworld("<think>inspect</think><action>look around</action>")
        self.assertEqual(reasoning, "inspect")
        self.assertEqual(action, "look around")

    def test_vwa_tier_b_and_three_tag_protocol(self):
        observation = "[12] [BUTTON] [Buy]\n[] [StaticText] [Hidden]\n[7] [LINK] [Home]"
        self.assertEqual(tier_b_observation(observation), "[12] [BUTTON]\n[7] [LINK]")
        parsed = parse_response(
            "<think>inspect</think><operation>click buy</operation><action>click [12]</action>"
        )
        self.assertTrue(parsed["parse_ok"])
        self.assertEqual(parsed["operation"], "click buy")
        self.assertFalse(parse_response("<think>inspect</think><action>click [12]</action>")["parse_ok"])

    def test_vwa_history_preserves_operation(self):
        history = format_history(
            [{"parse_ok": True, "operation": "open cart", "action": "click [3]"}]
        )
        self.assertEqual(history, "1. open cart -> click [3]")


if __name__ == "__main__":
    unittest.main()
