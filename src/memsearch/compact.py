"""Memory compact — compress and summarize chunks using pydantic-ai.

Default model chain: OpenAI Responses (`gpt-5.4`) → Gemini (`gemini-3.1-pro-preview`).
Credentials are read from provider-native env vars (`OPENAI_API_KEY`,
`GEMINI_API_KEY` or `GOOGLE_API_KEY`).
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel

COMPACT_PROMPT = """\
You are a knowledge compression assistant. Given the following chunks of text \
from a knowledge base, create a concise but comprehensive summary that preserves \
all key facts, decisions, code patterns, and actionable insights.

Chunks:
{chunks}

Write a clear, well-structured markdown summary. Use headings and bullet points. \
Preserve technical details, code snippets, and specific decisions."""

DEFAULT_MODELS: tuple[str, ...] = (
    "gpt-5.4",
    "gemini-3.1-pro-preview",
)


async def compact_chunks(
    chunks: list[dict[str, Any]],
    *,
    model: str | None = None,
    prompt_template: str | None = None,
) -> str:
    """Compress *chunks* into a summary using a pydantic-ai Agent.

    Parameters
    ----------
    chunks:
        List of chunk dicts (must contain ``"content"`` key).
    model:
        Single model identifier. When omitted, uses the ``DEFAULT_MODELS``
        fallback chain (OpenAI Responses → Gemini).
    prompt_template:
        Custom prompt template. Must contain ``{chunks}`` placeholder.
        Defaults to the built-in ``COMPACT_PROMPT``.
    """
    if not chunks:
        return ""
    combined = "\n\n---\n\n".join(c["content"] for c in chunks)
    template = prompt_template or COMPACT_PROMPT
    if "{chunks}" not in template:
        raise ValueError("prompt_template must include the {chunks} placeholder")
    prompt = template.format(chunks=combined)

    resolved = _resolve_model(model)
    agent = Agent(resolved)
    result = await agent.run(prompt)
    return result.output or ""


def _resolve_model(model: str | None) -> Model:
    if model:
        return _build_single_model(model)
    return _build_fallback_chain(DEFAULT_MODELS)


def _build_fallback_chain(names: tuple[str, ...]) -> Model:
    built = [_build_single_model(n) for n in names]
    if len(built) == 1:
        return built[0]
    return FallbackModel(built[0], *built[1:])


def _build_single_model(name: str) -> Model:
    if name.startswith("gemini-"):
        return GoogleModel(name)
    return OpenAIResponsesModel(name)
