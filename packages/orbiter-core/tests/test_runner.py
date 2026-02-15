"""Tests for orbiter.runner — public run() / run.sync() / run.stream() entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from orbiter.agent import Agent
from orbiter.runner import run
from orbiter.tool import tool
from orbiter.types import (
    AgentOutput,
    RunResult,
    StreamEvent,
    TextEvent,
    ToolCall,
    ToolCallEvent,
    Usage,
    UserMessage,
)

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


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


class _FakeStreamChunk:
    """Lightweight stream chunk for testing (mirrors StreamChunk fields)."""

    def __init__(
        self,
        delta: str = "",
        tool_call_deltas: list[Any] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.delta = delta
        self.tool_call_deltas = tool_call_deltas or []
        self.finish_reason = finish_reason
        self.usage = usage or Usage()


class _FakeToolCallDelta:
    """Mirrors ToolCallDelta fields for testing."""

    def __init__(
        self,
        index: int = 0,
        id: str | None = None,
        name: str | None = None,
        arguments: str = "",
    ) -> None:
        self.index = index
        self.id = id
        self.name = name
        self.arguments = arguments


def _make_stream_provider(
    stream_rounds: list[list[_FakeStreamChunk]],
) -> Any:
    """Create a mock provider with stream() returning pre-defined chunks.

    Each call to stream() consumes the next list of chunks from stream_rounds.
    """
    call_count = 0

    async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal call_count
        chunks = stream_rounds[min(call_count, len(stream_rounds) - 1)]
        call_count += 1
        for c in chunks:
            yield c

    mock = AsyncMock()
    mock.stream = stream
    # Also give it a complete() for agent._execute_tools (not needed for stream tests
    # but avoids AttributeError if somehow called)
    mock.complete = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# run.stream() tests
# ---------------------------------------------------------------------------


class TestRunStream:
    async def test_stream_text_events(self) -> None:
        """run.stream() yields TextEvent for each text delta."""
        agent = Agent(name="bot", instructions="Be nice.")
        chunks = [
            _FakeStreamChunk(delta="Hello"),
            _FakeStreamChunk(delta=" world"),
            _FakeStreamChunk(delta="!", finish_reason="stop"),
        ]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        async for ev in run.stream(agent, "Hi", provider=provider):
            events.append(ev)

        assert len(events) == 3
        assert all(isinstance(e, TextEvent) for e in events)
        assert events[0].text == "Hello"  # type: ignore[union-attr]
        assert events[1].text == " world"  # type: ignore[union-attr]
        assert events[2].text == "!"  # type: ignore[union-attr]

    async def test_stream_agent_name(self) -> None:
        """TextEvent includes agent_name."""
        agent = Agent(name="assistant")
        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "test", provider=provider)]

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].agent_name == "assistant"

    async def test_stream_empty_text(self) -> None:
        """Empty deltas are not yielded as events."""
        agent = Agent(name="bot")
        chunks = [
            _FakeStreamChunk(delta=""),
            _FakeStreamChunk(delta="data"),
            _FakeStreamChunk(delta=""),
        ]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "test", provider=provider)]

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].text == "data"


class TestRunStreamToolCalls:
    async def test_stream_tool_call_events(self) -> None:
        """run.stream() yields ToolCallEvent for tool calls."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hi {name}!"

        agent = Agent(name="bot", tools=[greet])

        # First round: tool call streamed via deltas
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="greet"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"name":'),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='"Alice"}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        # Second round: text response after tool execution
        round2 = [
            _FakeStreamChunk(delta="Done!"),
        ]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "Greet Alice", provider=provider)]

        # Should have: ToolCallEvent, then TextEvent("Done!")
        assert len(events) == 2
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].tool_name == "greet"
        assert events[0].tool_call_id == "tc1"
        assert events[0].agent_name == "bot"
        assert isinstance(events[1], TextEvent)
        assert events[1].text == "Done!"

    async def test_stream_multiple_tool_calls(self) -> None:
        """run.stream() handles multiple parallel tool calls."""

        @tool
        def add(a: int, b: int) -> str:
            """Add numbers."""
            return str(a + b)

        @tool
        def mul(a: int, b: int) -> str:
            """Multiply numbers."""
            return str(a * b)

        agent = Agent(name="calc", tools=[add, mul])

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="add"),
                    _FakeToolCallDelta(index=1, id="tc2", name="mul"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='{"a":2,"b":3}'),
                    _FakeToolCallDelta(index=1, arguments='{"a":4,"b":5}'),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Results ready.")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "compute", provider=provider)]

        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        text_events = [e for e in events if isinstance(e, TextEvent)]

        assert len(tool_events) == 2
        assert {e.tool_name for e in tool_events} == {"add", "mul"}
        assert len(text_events) == 1
        assert text_events[0].text == "Results ready."

    async def test_stream_tool_error_continues(self) -> None:
        """Tool execution errors don't crash the stream."""

        @tool
        def fail_tool() -> str:
            """Always fails."""
            raise ValueError("oops")

        agent = Agent(name="bot", tools=[fail_tool])

        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="fail_tool"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Tool failed.")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "try", provider=provider)]

        assert any(isinstance(e, ToolCallEvent) for e in events)
        assert any(isinstance(e, TextEvent) and e.text == "Tool failed." for e in events)


class TestRunStreamCompletion:
    async def test_stream_completes_on_text_only(self) -> None:
        """Stream ends after a text-only response (no tool calls)."""
        agent = Agent(name="bot")
        chunks = [
            _FakeStreamChunk(delta="All done."),
        ]
        provider = _make_stream_provider([chunks])

        events = [ev async for ev in run.stream(agent, "test", provider=provider)]

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)

    async def test_stream_no_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run.stream() raises when no provider is available."""
        from orbiter import runner as runner_mod
        from orbiter.agent import AgentError

        monkeypatch.setattr(runner_mod, "_resolve_provider", lambda _agent: None)
        agent = Agent(name="bot")

        with pytest.raises(AgentError, match="requires a provider"):
            async for _ in run.stream(agent, "test"):
                pass

    async def test_stream_max_steps(self) -> None:
        """Stream respects max_steps limit."""

        @tool
        def echo() -> str:
            """Echo."""
            return "echoed"

        agent = Agent(name="bot", tools=[echo], max_steps=2)

        # Every round returns tool calls — should stop after max_steps
        tool_round = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="echo"),
                ]
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        provider = _make_stream_provider([tool_round, tool_round, tool_round])

        events = [ev async for ev in run.stream(agent, "loop", provider=provider)]

        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        # Should have at most max_steps (2) tool call events
        assert len(tool_events) <= 2

    async def test_stream_text_and_tool_mixed(self) -> None:
        """Stream handles chunks with both text and tool call deltas."""

        @tool
        def compute() -> str:
            """Compute."""
            return "42"

        agent = Agent(name="bot", tools=[compute])

        round1 = [
            _FakeStreamChunk(
                delta="Thinking...",
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="compute"),
                ],
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments="{}"),
                ],
                finish_reason="tool_calls",
            ),
        ]
        round2 = [_FakeStreamChunk(delta="Answer: 42")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "what?", provider=provider)]

        types = [type(e).__name__ for e in events]
        assert "TextEvent" in types
        assert "ToolCallEvent" in types
