# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""WM-OPSD privileged self-distillation helpers.

The privileged teacher is the same policy weights conditioned on extra context:
the ground-truth next observation o' injected at the end of the user turn.
Only the student-generated ``<next_state>...</next_state>`` span is supervised.

Ported verbatim (semantics) from the validated slime hook
``slime/rollout/wm_privileged_opd.py``:
  - PRIVILEGED_HINT: the o' injection template.
  - build_privileged_prompt_ids: dataset-time — tokenize the templated prompt
    with the hint inserted before the assistant turn (response-independent).
  - next_state_span_mask: train-time — locate the <next_state> span in a freshly
    generated response via offset mapping (response only known post-rollout).
"""

NEXT_STATE_OPEN = "<next_state>"
NEXT_STATE_CLOSE = "</next_state>"

# Injected at the end of the user turn: shows the teacher the real next
# observation (which the student never sees).
PRIVILEGED_HINT = (
    "\n\n[Privileged information — ground truth] "
    "The actual resulting next observation is:\n{target_next_state}\n"
)

# Marker delimiting the end of the user turn in a Qwen2.5 chat template.
_ASSISTANT_MARKER = "<|im_end|>\n<|im_start|>assistant"


def build_privileged_prompt_ids(tokenizer, prompt_text, target_next_state):
    """Return token ids of the templated prompt with the privileged hint injected.

    ``prompt_text`` is the already chat-templated student prompt string (ending
    in ``...<|im_end|>\\n<|im_start|>assistant\\n``). The hint is inserted before
    the assistant turn so it becomes an extra known condition for the world model
    (not part of the assistant output), exactly as in the slime hook.
    """
    hint = PRIVILEGED_HINT.format(target_next_state=target_next_state)
    idx = prompt_text.rfind(_ASSISTANT_MARKER)
    if idx != -1:
        privileged_text = prompt_text[:idx] + hint + prompt_text[idx:]
    else:
        # Fallback (should not happen with a well-formed chat template).
        privileged_text = prompt_text + hint
    return tokenizer.encode(privileged_text, add_special_tokens=False)


def next_state_span_mask(tokenizer, response_ids):
    """Locate <next_state>...</next_state> in a response; return a 0/1 mask.

    ``<next_state>`` tokenizes into several tokens, so we can't match by token-id
    substring. Instead we decode to text and use the fast tokenizer's offset
    mapping to map the character span back to token indices, then align to the
    length of ``response_ids``.

    Returns (mask: list[int] of len==len(response_ids), ok: bool). ok=False means
    no valid span was found — the caller should drop / zero-mask that sample.
    """
    text = tokenizer.decode(response_ids)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    open_char = text.find(NEXT_STATE_OPEN)
    if open_char == -1:
        return [0] * len(response_ids), False
    span_start_char = open_char + len(NEXT_STATE_OPEN)
    close_char = text.find(NEXT_STATE_CLOSE, span_start_char)
    # Missing </next_state>: take to end (model didn't close the tag but the
    # content is still next-state).
    span_end_char = close_char if close_char != -1 else len(text)
    if span_end_char <= span_start_char:
        return [0] * len(response_ids), False

    # Use "any overlap" (not strict containment) so tokens straddling a span
    # boundary (e.g. '>You', '.</') still count as in-span.
    mask_by_reenc = [
        1 if (o[0] is not None and o[1] > o[0] and o[1] > span_start_char and o[0] < span_end_char) else 0
        for o in offsets
    ]
    n = len(response_ids)
    if len(mask_by_reenc) >= n:
        mask = mask_by_reenc[:n]
    else:
        mask = mask_by_reenc + [0] * (n - len(mask_by_reenc))
    return mask, any(mask)
