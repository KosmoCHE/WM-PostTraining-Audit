"""ScienceWorld no-list ReAct prompt and strict action extraction."""

from __future__ import annotations

import re


SCIWORLD_ACTION_TYPES = """\
activate (object): turn on / activate an object
close (object): close a container or door
connect (object) to (object): connect one electrical terminal to another
deactivate (object): turn off / deactivate an object
disconnect (object): disconnect an electrical terminal
dunk (object) in (object): dunk an object into a container of liquid
eat (object): eat an object
flush (object): flush an object
focus on (object): focus on the task-relevant object
go (location): move to a connected location or room
inventory: list what you are carrying
look around: describe the current room
look at (object): examine an object
look in (object): look inside a container
mix (object): mix the contents of a container
move (object) to (object): move an object to a container or receptacle
open (object): open a container or door
pick up (object): pick up an object into your inventory
pour (object) in (object): pour a container's contents into another container
put down (object): drop the held object
read (object): read text on an object
task: print the current task description
use (object) on (object): use one object on another
wait: wait for up to 10 iterations
wait1: wait a single iteration"""


VARIANT_NO_LIST = """\
You are an expert agent operating in the ScienceWorld text environment. Your task is to: {task_description}
Here are the types of actions you can take:
{action_types}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an action for current step and present it within <action> </action> tags.
"""


def build_action_history(transitions: list[dict]) -> str:
    """Format complete observation/action history without a sliding window."""
    if not transitions:
        return "N/A (this is the first step)."
    lines = []
    for index, transition in enumerate(transitions, 1):
        lines.append(f"Observation {index}: {transition['observation']}")
        action = transition["action"] or "(no valid action)"
        lines.append(f"Action {index}: {action}")
    return "\n".join(lines)


def render_prompt(
    task: str, step_index: int, transitions: list[dict], current_observation: str
) -> str:
    """Render the ScienceWorld no-list prompt for one interaction step."""
    return VARIANT_NO_LIST.format(
        task_description=task,
        action_types=SCIWORLD_ACTION_TYPES,
        step_count=len(transitions),
        history_length=len(transitions),
        action_history=build_action_history(transitions),
        current_step=step_index,
        current_observation=current_observation,
    )


def parse_output(content: str) -> tuple[str | None, str | None]:
    """Extract ``<think>`` and ``<action>`` spans with an Action-line fallback."""
    think_match = re.search(r"<think>(.*?)</think>", content, re.S | re.I)
    action_match = re.search(r"<action>(.*?)</action>", content, re.S | re.I)
    if not action_match:
        action_match = re.search(r"^Action:\s*(.+)$", content, re.M | re.I)
    reasoning = think_match.group(1).strip() if think_match else None
    action = action_match.group(1).strip() if action_match else None
    return reasoning, action
