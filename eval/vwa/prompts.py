"""VisualWebArena tier-B prompt and strict three-tag response parser."""

from __future__ import annotations

import re


_SOM_LINE = re.compile(r"^\[(\d*)\] \[([^\]]*)\] \[(.*)\]$")
_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_OPERATION = re.compile(r"<operation>(.*?)</operation>", re.S)
_ACTION = re.compile(r"<action>(.*?)</action>", re.S)


ACTION_SPACE = """The actions you can perform fall into several categories:

Page Operation Actions:
```click [id]```: This action clicks on an element with a specific id on the webpage.
```type [id] [content]```: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., ```type [id] [content] [0]```.
```hover [id]```: Hover over an element with id.
```press [key_comb]```: Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
```scroll [down]``` or ```scroll [up]```: Scroll the page up or down.

Tab Management Actions:
```new_tab```: Open a new, empty browser tab.
```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.
```close_tab```: Close the currently active tab.

URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If the objective doesn't require an answer, use ```stop []```."""


OBSERVATION_DESCRIPTION = """Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The observation: a text list of the interactable element ids and types on the current page, as [id] [tagType]. The text content of elements is NOT included — read it from the screenshot.
The current page screenshot: each interactable element is annotated with a numbered bounding box; the number matches the id in the observation.
The current web page's URL, and your action history so far."""


OUTPUT_FORMAT = """Your response must strictly follow this format (all three tags, in this order):
<think>Your step-by-step reasoning about the current state and what to do next.</think>
<operation>A one-sentence description of the operation you are about to perform.</operation>
<action>The action command, e.g. click [45]</action>

Rules:
1. Issue exactly one action per response, and it must be valid in the current page.
2. The action command must exactly follow the documented format. Any malformed output counts as a failure.
3. Issue the stop action when you believe the objective is achieved. Do not generate anything after the action tag."""


SYSTEM_PROMPT = (
    "You are an autonomous intelligent agent tasked with navigating a web browser. "
    "You will be given web-based tasks. These tasks will be accomplished through the "
    "use of specific actions you can issue.\n\n"
    f"{OBSERVATION_DESCRIPTION}\n\n{ACTION_SPACE}\n\n{OUTPUT_FORMAT}"
)


def tier_b_observation(som_text: str) -> str:
    """Keep interactable element IDs and tag types while removing element text."""
    lines = []
    for line in som_text.splitlines():
        match = _SOM_LINE.match(line.strip())
        if not match:
            continue
        element_id, tag, _ = match.groups()
        if element_id:
            lines.append(f"[{element_id}] [{tag}]")
    return "\n".join(lines)


def format_history(steps: list[dict]) -> str:
    """Format prior operations and actions exactly as used by the paper run."""
    if not steps:
        return "None (this is the first step)."
    lines = []
    for index, step in enumerate(steps, 1):
        if not step.get("parse_ok"):
            lines.append(f"{index}. <invalid output>")
            continue
        operation = (step.get("operation") or "").strip().replace("\n", " ")
        action = (step.get("action") or "").strip()
        lines.append(f"{index}. {operation} -> {action}" if operation else f"{index}. {action}")
    return "\n".join(lines)


def build_user_text(objective: str, history: str, observation: str, url: str) -> str:
    """Build the text portion of the multimodal user message."""
    return "\n\n".join(
        [
            f"OBJECTIVE: {objective}",
            f"ACTION HISTORY:\n{history}",
            f"OBSERVATION:\n{observation}",
            f"URL: {url}",
            "CURRENT PAGE: (see the screenshot below)",
        ]
    )


def parse_response(text: str) -> dict:
    """Require think, operation, and action tags in that order."""
    think = _THINK.search(text)
    operation = _OPERATION.search(text)
    action = _ACTION.search(text)
    valid = bool(
        think
        and operation
        and action
        and think.start() < operation.start() < action.start()
    )
    return {
        "parse_ok": valid,
        "think": think.group(1).strip() if think else None,
        "operation": operation.group(1).strip() if operation else None,
        "action": action.group(1).strip().strip("`").strip() if action else None,
    }
