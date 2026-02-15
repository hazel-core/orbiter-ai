"""Tests for orbiter.runner — public run() / run.sync() entry point."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from orbiter.agent import Agent
from orbiter.runner import run
from orbiter.tool import tool
from orbiter.types import AgentOutput, RunResult, ToolCall, Usage, UserMessage

# ---------------------------------------------------------------------------
# Fixtures: mock provider
# ---------------------------------------------------------------------------


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Create a mock provider returning pre-defined AgentOutput values."""
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        class FakeResponse:
            content = resp.text
            tool_calls = resp.tool_calls
            usage = resp.usage

        return FakeResponse()

    mock = AsyncMock()
    mock.complete = complete
    return mock


# ---------------------------------------------------------------------------
# run() async
# ---------------------------------------------------------------------------


class TestRunAsync:
    async def test_basic_run(self) -> None:
        """run() returns RunResult with agent output."""
        agent = Agent(name="bot", instructions="Be helpful.")
        provider = _make_provider([AgentOutput(text="Hello!")])

        result = await run(agent, "Hi", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "Hello!"
        assert result.steps >= 1

    async def test_run_with_usage(self) -> None:
        """run() propagates token usage to RunResult."""
        agent = Agent(name="bot")
        usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        provider = _make_provider([AgentOutput(text="ok", usage=usage)])

        result = await run(agent, "test", provider=provider)

        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150

    async def test_run_includes_messages(self) -> None:
        """run() result includes message history."""
        agent = Agent(name="bot", instructions="You are nice.")
        provider = _make_provider([AgentOutput(text="Sure!")])

        result = await run(agent, "hello", provider=provider)

        assert len(result.messages) > 0

    async def test_run_with_tool(self) -> None:
        """run() handles agents with tools (tool call → text response)."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="greeter", tools=[greet])
        responses = [
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="greet", arguments='{"name":"Alice"}')],
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
            AgentOutput(
                text="I greeted Alice for you!",
                usage=Usage(input_tokens=30, output_tokens=15, total_tokens=45),
            ),
        ]
        provider = _make_provider(responses)

        result = await run(agent, "Greet Alice", provider=provider)

        assert result.output == "I greeted Alice for you!"


# ---------------------------------------------------------------------------
# run.sync()
# ---------------------------------------------------------------------------


class TestRunSync:
    def test_sync_basic(self) -> None:
        """run.sync() returns RunResult synchronously."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="sync ok")])

        result = run.sync(agent, "test", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "sync ok"

    def test_sync_with_usage(self) -> None:
        """run.sync() propagates usage stats."""
        agent = Agent(name="bot")
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        provider = _make_provider([AgentOutput(text="ok", usage=usage)])

        result = run.sync(agent, "test", provider=provider)

        assert result.usage.input_tokens == 10

    def test_sync_with_tool(self) -> None:
        """run.sync() handles tool-calling agents."""

        @tool
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        agent = Agent(name="calc", tools=[add])
        responses = [
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="add", arguments='{"a":2,"b":3}')],
            ),
            AgentOutput(text="The answer is 5."),
        ]
        provider = _make_provider(responses)

        result = run.sync(agent, "What is 2+3?", provider=provider)

        assert result.output == "The answer is 5."


# ---------------------------------------------------------------------------
# Multi-turn via messages param
# ---------------------------------------------------------------------------


class TestRunMultiTurn:
    async def test_prior_messages(self) -> None:
        """run() forwards prior messages for multi-turn conversations."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="Continued!")])

        prior = [UserMessage(content="Earlier message")]
        result = await run(agent, "Continue", messages=prior, provider=provider)

        assert result.output == "Continued!"

    async def test_multi_turn_accumulates(self) -> None:
        """Multiple run() calls with messages param create ongoing conversation."""
        agent = Agent(name="bot", instructions="You are a counter.")
        provider1 = _make_provider([AgentOutput(text="Count: 1")])

        r1 = await run(agent, "Start counting", provider=provider1)
        assert r1.output == "Count: 1"

        # Second turn passes first result's messages
        provider2 = _make_provider([AgentOutput(text="Count: 2")])
        r2 = await run(agent, "Next", messages=r1.messages, provider=provider2)
        assert r2.output == "Count: 2"


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestRunErrors:
    async def test_no_provider_raises(self) -> None:
        """run() raises when no provider given and auto-resolve fails."""
        agent = Agent(name="bot")

        # Agent.run() raises AgentError → call_runner wraps in CallRunnerError
        from orbiter._internal.call_runner import CallRunnerError

        with pytest.raises(CallRunnerError):
            await run(agent, "test")

    async def test_provider_error_propagates(self) -> None:
        """Provider errors bubble up through run()."""
        agent = Agent(name="bot")
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("Service down"))

        from orbiter._internal.call_runner import CallRunnerError

        with pytest.raises(CallRunnerError, match="Call runner failed"):
            await run(agent, "test", provider=provider)

    def test_sync_error_propagates(self) -> None:
        """run.sync() propagates errors from the async path."""
        agent = Agent(name="bot")

        from orbiter._internal.call_runner import CallRunnerError

        with pytest.raises(CallRunnerError):
            run.sync(agent, "test")


# ---------------------------------------------------------------------------
# Provider auto-resolution
# ---------------------------------------------------------------------------


class TestProviderAutoResolve:
    async def test_explicit_provider_used(self) -> None:
        """When provider is given explicitly, it is used directly."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="explicit")])

        result = await run(agent, "test", provider=provider)

        assert result.output == "explicit"
