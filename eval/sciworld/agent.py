"""ScienceWorld ReAct agent backed by an OpenAI-compatible model server."""

from __future__ import annotations

from eval.common.client import chat, make_client
from eval.sciworld.prompts import parse_output, render_prompt


class ReActAgent:
    """Maintain one episode's no-list interaction history."""

    def __init__(
        self, base_url: str, model: str, temperature: float, max_tokens: int
    ):
        self.client = make_client(base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.task = ""
        self.transitions: list[dict] = []

    def act(self, step_index: int, observation: str) -> tuple[str | None, str | None]:
        """Generate reasoning and one action for the current observation."""
        prompt = render_prompt(self.task, step_index, self.transitions, observation)
        content, _ = chat(
            self.client,
            self.model,
            [{"role": "user", "content": prompt}],
            self.temperature,
            self.max_tokens,
        )
        return parse_output(content)

    def record(self, observation: str, action: str | None, next_observation: str):
        """Append one completed transition to the prompt history."""
        self.transitions.append(
            {
                "observation": observation,
                "action": action,
                "next_observation": next_observation,
            }
        )
