"""ReAct agent backed by an OpenAI-compatible model server."""

from __future__ import annotations

from eval.alfworld.prompts import parse_output, render_prompt
from eval.common.client import chat, make_client


class ReActAgent:
    """Maintain one episode's history and request one action per step."""

    def __init__(
        self,
        base_url: str,
        variant: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        self.client = make_client(base_url)
        self.variant = variant
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.task = ""
        self.transitions: list[dict] = []

    def act(
        self, step_index: int, observation: str, admissible_actions: list[str]
    ) -> tuple[str, str | None, int | None]:
        """Generate reasoning and one action for the current observation."""
        prompt = render_prompt(
            self.variant,
            self.task,
            step_index,
            self.transitions,
            observation,
            admissible_actions,
        )
        content, tokens = chat(
            self.client,
            self.model,
            [{"role": "user", "content": prompt}],
            self.temperature,
            self.max_tokens,
        )
        reasoning, action = parse_output(content)
        return reasoning, action, tokens

    def record(self, observation: str, action: str | None, next_observation: str):
        """Append one completed transition to the prompt history."""
        self.transitions.append(
            {
                "observation": observation,
                "action": action,
                "next_observation": next_observation,
            }
        )
