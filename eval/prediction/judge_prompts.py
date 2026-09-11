"""Dynamics-focused judge prompts for next-observation prediction."""


ALFWORLD_JUDGE_PROMPT = """\
You are an expert evaluator of world-model predictions for the ALFRED (ALFWorld) text environment.

Given the current observation and an action, an agent predicted the resulting next observation. \
You are given the GROUND-TRUTH next observation actually returned by the environment. Judge whether \
the agent's prediction is CORRECT about the EFFECT OF THE ACTION.

## Task the agent is doing
{task}

## Current observation
{current_obs}

## Action taken
{action}

## Ground-truth next observation (from the environment)
{ground_truth}

## Agent's predicted next observation
{prediction}

## How to judge
Judge whether the prediction captures the DYNAMICS — what this action does to the world state — \
matching the ground truth.

Rules:
1. IGNORE surface differences: wording, paraphrase, formatting, ordering, length, extra reasoning. \
The agent writes in its own words and does NOT copy the environment's phrasing. Judge meaning, not \
string overlap.
2. Judge the ACTION'S EFFECT: did it get right what the action does — arriving at a location, a \
receptacle being open/closed, picking up / putting an object, an action succeeding vs. failing \
("Nothing happens."), a state change (heated/cooled/cleaned/sliced)?
3. Unpredictable specifics are NOT required: content only revealed by doing the action and not \
inferable beforehand — e.g. exactly which objects are on a surface you just walked to, or inside a \
container you just opened — does NOT need to match. Do not penalize different/guessed specific objects.
4. HALLUCINATION / OVER-PREDICTION is wrong: mark INCORRECT if the prediction adds events that did \
not happen (e.g. the action was only "go to X" but the prediction also opens it and lists contents), \
predicts success when the action failed (or vice versa), a wrong resulting location/state, or \
otherwise contradicts the ground-truth dynamics.

Verdict:
- CORRECT: dynamically consistent with the ground truth (right effect of the action), ignoring \
wording and unpredictable specifics, and without hallucinating events that did not occur.
- INCORRECT: wrong effect, contradicts the ground truth, or hallucinates/over-predicts.

First give a one-sentence justification. Then output the verdict on its own final line, exactly one of:
VERDICT: CORRECT
VERDICT: INCORRECT
"""


SCIWORLD_JUDGE_PROMPT = """\
You are an expert evaluator of world-model predictions for the ScienceWorld text environment.

Given the current observation and an action, an agent predicted the resulting next observation. \
You are given the GROUND-TRUTH next observation actually returned by the environment. Judge whether \
the agent's prediction is CORRECT about the EFFECT OF THE ACTION.

## Task the agent is doing
{task}

## Current observation
{current_obs}

## Action taken
{action}

## Ground-truth next observation (from the environment)
{ground_truth}

## Agent's predicted next observation
{prediction}

## How to judge
Judge whether the prediction captures the DYNAMICS — what this action does to the world state — \
matching the ground truth.

Rules:
1. IGNORE surface differences: wording, paraphrase, formatting, ordering, length, extra reasoning. \
The agent writes in its own words and does NOT copy the environment's phrasing. Judge meaning, not \
string overlap.
2. Judge the ACTION'S EFFECT: did it get right what the action does — moving to a location, a \
door/container being opened/closed, picking up / moving an object, activating/deactivating a device, \
focusing on an object, an action succeeding vs. failing ("No known action matches that input." / \
"The door is not open."), or a physical state change (heated/cooled/melted/frozen/boiled/mixed)?
3. Unpredictable specifics are NOT required: content only revealed by doing the action and not \
inferable beforehand — e.g. exactly which objects are in a room you just entered, inside a container \
you just opened, or a measured numeric value (temperature/melting point) — does NOT need to match. \
Do not penalize different/guessed specific objects or numbers.
4. HALLUCINATION / OVER-PREDICTION is wrong: mark INCORRECT if the prediction adds events that did \
not happen (e.g. the action was only "go to X" but the prediction also opens a container and lists \
contents), predicts success when the action failed (or vice versa), a wrong resulting location/state, \
or otherwise contradicts the ground-truth dynamics.

Verdict:
- CORRECT: dynamically consistent with the ground truth (right effect of the action), ignoring \
wording and unpredictable specifics, and without hallucinating events that did not occur.
- INCORRECT: wrong effect, contradicts the ground truth, or hallucinates/over-predicts.

First give a one-sentence justification. Then output the verdict on its own final line, exactly one of:
VERDICT: CORRECT
VERDICT: INCORRECT
"""


JUDGE_PROMPTS = {
    "alfworld": ALFWORLD_JUDGE_PROMPT,
    "sciworld": SCIWORLD_JUDGE_PROMPT,
}
