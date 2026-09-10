"""Prediction prompt used to construct ScienceWorld training samples."""

WM_PROMPT_TEMPLATE = """\
You are a world model for the ScienceWorld text environment. Given the interaction history, the current observation, and the action just taken, predict the resulting next observation.

Task: {task}

Interaction history:
{history}

Current observation: {current_observation}
Action taken: {action}

First reason step-by-step about the effect of the action within <think> </think> tags. Then give the predicted next observation within <next_state> </next_state> tags.
"""
