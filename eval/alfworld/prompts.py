"""ALFWorld ReAct prompts and strict action extraction."""

import re


VARIANT_GIVE_LIST = """\
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


VARIANT_NO_LIST = """\
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Here are the actions you can take:
go to (receptacle): move to a receptacle
open (receptacle): open a receptacle
close (receptacle): close a receptacle
take (object) from (receptacle): take an object from a receptacle
put (object) in/on (receptacle): place an object in or on a receptacle
examine (something): examine a receptacle or an object
use (object): use an object
heat (object) with (receptacle): heat an object using a receptacle
clean (object) with (receptacle): clean an object using a receptacle
cool (object) with (receptacle): cool an object using a receptacle
slice (object) with (object): slice an object using a sharp object
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an action for current step and present it within <action> </action> tags.
"""


_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.S)
_ACTION_FALLBACK_RE = re.compile(
    r"(?im)^\s*\*{0,2}\s*Action\s*:?\s*\*{0,2}\s*(.+?)\s*$"
)


def build_action_history(transitions: list[dict]) -> str:
    """Format the complete observation/action history without a sliding window."""
    if not transitions:
        return "N/A (this is the first step)."
    lines = []
    for index, transition in enumerate(transitions, 1):
        lines.append(f"Observation {index}: {transition['observation']}")
        lines.append(f"Action {index}: {transition['action']}")
    return "\n".join(lines)


def render_prompt(
    variant: str,
    task: str,
    step_index: int,
    transitions: list[dict],
    current_observation: str,
    admissible_actions: list[str],
) -> str:
    """Render the list or no-list prompt for one interaction step."""
    fields = {
        "task_description": task,
        "step_count": len(transitions),
        "history_length": len(transitions),
        "action_history": build_action_history(transitions),
        "current_step": step_index,
        "current_observation": current_observation,
    }
    if variant == "give":
        return VARIANT_GIVE_LIST.format(
            admissible_actions=", ".join(admissible_actions), **fields
        )
    if variant == "nolist":
        return VARIANT_NO_LIST.format(**fields)
    raise ValueError(f"unknown ALFWorld action regime: {variant}")


def parse_output(content: str) -> tuple[str, str | None]:
    """Extract reasoning and one action; return ``None`` for malformed output."""
    tags = _ACTION_TAG_RE.findall(content)
    if tags:
        return _ACTION_TAG_RE.sub("", content).strip(), tags[-1].strip()
    match = _ACTION_FALLBACK_RE.search(content)
    if match:
        return content[: match.start()].strip(), match.group(1).strip()
    return content.strip(), None
