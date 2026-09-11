"""OpenAI-compatible client helpers shared by evaluation runners."""

from __future__ import annotations

import os


def make_client(base_url: str, timeout: float = 180.0):
    """Create a client for a server root URL that does not include ``/v1``."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "placeholder")
    return OpenAI(
        base_url=base_url.rstrip("/") + "/v1",
        api_key=api_key,
        timeout=timeout,
        max_retries=4,
    )


def chat(
    client,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    top_p: float | None = None,
) -> tuple[str, int | None]:
    """Generate one response and return its text plus completion-token count."""
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    tokens = response.usage.completion_tokens if response.usage else None
    return text, tokens
