from __future__ import annotations

from dataclasses import dataclass

import pytest

from memsearch import compact as compact_module


@pytest.mark.asyncio
async def test_compact_chunks_returns_empty_string_for_empty_input() -> None:
    assert await compact_module.compact_chunks([]) == ""


@pytest.mark.asyncio
async def test_compact_chunks_runs_agent_with_default_chain(monkeypatch) -> None:
    captured: dict[str, object] = {}

    @dataclass
    class _FakeResult:
        output: str

    class _FakeAgent:
        def __init__(self, model):
            captured["model"] = model

        async def run(self, prompt):
            captured["prompt"] = prompt
            return _FakeResult(output="agent-summary")

    monkeypatch.setattr(compact_module, "Agent", _FakeAgent)

    def fake_chain(names):
        captured["fallback_names"] = tuple(names)
        return "fake-fallback-model"

    monkeypatch.setattr(compact_module, "_build_fallback_chain", fake_chain)

    result = await compact_module.compact_chunks(
        [{"content": "alpha"}, {"content": "beta"}],
    )

    assert result == "agent-summary"
    assert captured["model"] == "fake-fallback-model"
    assert captured["fallback_names"] == compact_module.DEFAULT_MODELS
    assert captured["prompt"] == compact_module.COMPACT_PROMPT.format(chunks="alpha\n\n---\n\nbeta")


@pytest.mark.asyncio
async def test_compact_chunks_uses_single_model_override(monkeypatch) -> None:
    captured: dict[str, object] = {}

    @dataclass
    class _FakeResult:
        output: str

    class _FakeAgent:
        def __init__(self, model):
            captured["model"] = model

        async def run(self, prompt):
            captured["prompt"] = prompt
            return _FakeResult(output="single-summary")

    monkeypatch.setattr(compact_module, "Agent", _FakeAgent)

    def fake_single(name):
        captured["single_name"] = name
        return f"built:{name}"

    monkeypatch.setattr(compact_module, "_build_single_model", fake_single)

    result = await compact_module.compact_chunks(
        [{"content": "memory chunk"}],
        model="gpt-5.4",
        prompt_template="Summarize:\n{chunks}",
    )

    assert result == "single-summary"
    assert captured["model"] == "built:gpt-5.4"
    assert captured["single_name"] == "gpt-5.4"
    assert captured["prompt"] == "Summarize:\nmemory chunk"


@pytest.mark.asyncio
async def test_compact_chunks_rejects_prompt_without_chunks_placeholder() -> None:
    with pytest.raises(ValueError, match=r"prompt_template must include the \{chunks\} placeholder"):
        await compact_module.compact_chunks(
            [{"content": "x"}],
            prompt_template="Summarize the memory carefully.",
        )


def test_build_single_model_routes_gemini_to_google(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeGoogle:
        def __init__(self, name):
            captured["google"] = name

    class _FakeOpenAI:
        def __init__(self, name):
            captured["openai"] = name

    monkeypatch.setattr(compact_module, "GoogleModel", _FakeGoogle)
    monkeypatch.setattr(compact_module, "OpenAIResponsesModel", _FakeOpenAI)

    compact_module._build_single_model("gemini-3.1-pro-preview")
    compact_module._build_single_model("gpt-5.4")

    assert captured == {"google": "gemini-3.1-pro-preview", "openai": "gpt-5.4"}
