"""Tests for orbiter._internal.handlers — Handler ABC and AgentHandler."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from orbiter._internal.handlers import (
    AgentHandler,
    Handler,
    HandlerError,
    SwarmMode,
)
from orbiter.agent import Agent
from orbiter.types import AgentOutput, RunResult

# ---------------------------------------------------------------------------
# Fixtures: mock provider
# ---------------------------------------------------------------------------


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Create a mock provider returning pre-defined AgentOutput values.

    Supports multiple agents by cycling through responses in call order.
    """
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
# Handler ABC
# ---------------------------------------------------------------------------


class TestHandlerABC:
    def test_handler_is_abstract(self) -> None:
        """Handler cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Handler()  # type: ignore[abstract]

    async def test_concrete_handler(self) -> None:
        """A concrete Handler subclass can yield results."""

        class EchoHandler(Handler[str, str]):
            async def handle(self, input: str, **kwargs: Any) -> AsyncIterator[str]:
                yield f"echo: {input}"

        handler = EchoHandler()
        results = [r async for r in handler.handle("hello")]

        assert results == ["echo: hello"]

    async def test_handler_multiple_yields(self) -> None:
        """Handler can yield multiple outputs."""

        class SplitHandler(Handler[str, str]):
            async def handle(self, input: str, **kwargs: Any) -> AsyncIterator[str]:
                for word in input.split():
                    yield word

        handler = SplitHandler()
        results = [r async for r in handler.handle("a b c")]

        assert results == ["a", "b", "c"]

    async def test_handler_empty_yield(self) -> None:
        """Handler can yield nothing."""

        class EmptyHandler(Handler[str, str]):
            async def handle(self, input: str, **kwargs: Any) -> AsyncIterator[str]:
                return
                yield

        handler = EmptyHandler()
        results = [r async for r in handler.handle("anything")]

        assert results == []


# ---------------------------------------------------------------------------
# SwarmMode enum
# ---------------------------------------------------------------------------


class TestSwarmMode:
    def test_modes(self) -> None:
        """SwarmMode has workflow, handoff, and team values."""
        assert SwarmMode.WORKFLOW == "workflow"
        assert SwarmMode.HANDOFF == "handoff"
        assert SwarmMode.TEAM == "team"

    def test_mode_from_string(self) -> None:
        """SwarmMode can be created from string value."""
        assert SwarmMode("workflow") == SwarmMode.WORKFLOW


# ---------------------------------------------------------------------------
# AgentHandler — workflow mode
# ---------------------------------------------------------------------------


class TestAgentHandlerWorkflow:
    async def test_single_agent_workflow(self) -> None:
        """Workflow with one agent returns its result."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="Hello!")])

        handler = AgentHandler(
            agents={"bot": agent},
            mode=SwarmMode.WORKFLOW,
            flow_order=["bot"],
            provider=provider,
        )
        results = [r async for r in handler.handle("Hi")]

        assert len(results) == 1
        assert results[0].output == "Hello!"

    async def test_multi_agent_workflow(self) -> None:
        """Workflow runs agents in order, chaining output as input."""
        agent_a = Agent(name="agent_a")
        agent_b = Agent(name="agent_b")
        # agent_a outputs "Step 1", agent_b outputs "Step 2"
        provider = _make_provider(
            [
                AgentOutput(text="Step 1"),
                AgentOutput(text="Step 2"),
            ]
        )

        handler = AgentHandler(
            agents={"agent_a": agent_a, "agent_b": agent_b},
            mode=SwarmMode.WORKFLOW,
            flow_order=["agent_a", "agent_b"],
            provider=provider,
        )
        results = [r async for r in handler.handle("Start")]

        assert len(results) == 2
        assert results[0].output == "Step 1"
        assert results[1].output == "Step 2"

    async def test_three_agent_pipeline(self) -> None:
        """Workflow correctly chains 3 agents."""
        agents = {f"a{i}": Agent(name=f"a{i}") for i in range(3)}
        provider = _make_provider(
            [
                AgentOutput(text="out_0"),
                AgentOutput(text="out_1"),
                AgentOutput(text="out_2"),
            ]
        )

        handler = AgentHandler(
            agents=agents,
            mode=SwarmMode.WORKFLOW,
            flow_order=["a0", "a1", "a2"],
            provider=provider,
        )
        results = [r async for r in handler.handle("input")]

        assert len(results) == 3
        assert [r.output for r in results] == ["out_0", "out_1", "out_2"]

    async def test_workflow_missing_agent_raises(self) -> None:
        """Workflow raises HandlerError for missing agent."""
        handler = AgentHandler(
            agents={},
            mode=SwarmMode.WORKFLOW,
            flow_order=["missing"],
            provider=_make_provider([]),
        )

        with pytest.raises(HandlerError, match="not found"):
            async for _ in handler.handle("test"):
                pass


# ---------------------------------------------------------------------------
# AgentHandler — handoff mode
# ---------------------------------------------------------------------------


class TestAgentHandlerHandoff:
    async def test_no_handoff(self) -> None:
        """Agent without handoff targets terminates after first run."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="Done")])

        handler = AgentHandler(
            agents={"bot": agent},
            mode=SwarmMode.HANDOFF,
            flow_order=["bot"],
            provider=provider,
        )
        results = [r async for r in handler.handle("Hi")]

        assert len(results) == 1
        assert results[0].output == "Done"

    async def test_handoff_chain(self) -> None:
        """Agent A hands off to Agent B by outputting target name."""
        agent_b = Agent(name="billing")
        agent_a = Agent(name="triage", handoffs=[agent_b])

        # triage outputs "billing" (exact match → handoff), billing outputs "Done"
        provider = _make_provider(
            [
                AgentOutput(text="billing"),
                AgentOutput(text="Billing handled."),
            ]
        )

        handler = AgentHandler(
            agents={"triage": agent_a, "billing": agent_b},
            mode=SwarmMode.HANDOFF,
            flow_order=["triage"],
            provider=provider,
        )
        results = [r async for r in handler.handle("Help me")]

        assert len(results) == 2
        assert results[0].output == "billing"
        assert results[1].output == "Billing handled."

    async def test_handoff_max_exceeded(self) -> None:
        """Exceeding max_handoffs raises HandlerError."""
        agent_b = Agent(name="b")
        agent_a = Agent(name="a", handoffs=[agent_b])
        # b hands off back to a
        agent_b_inst = Agent(name="b", handoffs=[agent_a])

        # Alternate: a outputs "b", b outputs "a", a outputs "b", ...
        provider = _make_provider(
            [
                AgentOutput(text="b"),
                AgentOutput(text="a"),
                AgentOutput(text="b"),
                AgentOutput(text="a"),
                AgentOutput(text="b"),
            ]
        )

        handler = AgentHandler(
            agents={"a": agent_a, "b": agent_b_inst},
            mode=SwarmMode.HANDOFF,
            flow_order=["a"],
            provider=provider,
            max_handoffs=3,
        )

        with pytest.raises(HandlerError, match="Max handoffs"):
            async for _ in handler.handle("test"):
                pass

    async def test_handoff_detection_exact_match(self) -> None:
        """Handoff detection requires exact match of output to target name."""
        target = Agent(name="support")
        agent = Agent(name="triage", handoffs=[target])
        # Output contains "support" but isn't exactly "support"
        provider = _make_provider([AgentOutput(text="Contact support please")])

        handler = AgentHandler(
            agents={"triage": agent, "support": target},
            mode=SwarmMode.HANDOFF,
            flow_order=["triage"],
            provider=provider,
        )
        results = [r async for r in handler.handle("help")]

        # No handoff — output doesn't exactly match target name
        assert len(results) == 1

    async def test_handoff_missing_agent_raises(self) -> None:
        """Handoff to missing agent raises HandlerError."""
        handler = AgentHandler(
            agents={},
            mode=SwarmMode.HANDOFF,
            flow_order=["missing"],
            provider=_make_provider([]),
        )

        with pytest.raises(HandlerError, match="not found"):
            async for _ in handler.handle("test"):
                pass


# ---------------------------------------------------------------------------
# AgentHandler — team mode
# ---------------------------------------------------------------------------


class TestAgentHandlerTeam:
    async def test_team_lead_runs(self) -> None:
        """Team mode runs the lead agent."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        provider = _make_provider([AgentOutput(text="Lead result")])

        handler = AgentHandler(
            agents={"lead": lead, "worker": worker},
            mode=SwarmMode.TEAM,
            flow_order=["lead", "worker"],
            provider=provider,
        )
        results = [r async for r in handler.handle("coordinate")]

        assert len(results) == 1
        assert results[0].output == "Lead result"

    async def test_team_empty_flow_raises(self) -> None:
        """Team mode with no agents raises HandlerError."""
        handler = AgentHandler(
            agents={},
            mode=SwarmMode.TEAM,
            flow_order=[],
            provider=_make_provider([]),
        )

        with pytest.raises(HandlerError, match="requires at least one agent"):
            async for _ in handler.handle("test"):
                pass

    async def test_team_missing_lead_raises(self) -> None:
        """Team mode with missing lead agent raises HandlerError."""
        handler = AgentHandler(
            agents={"worker": Agent(name="worker")},
            mode=SwarmMode.TEAM,
            flow_order=["missing_lead"],
            provider=_make_provider([]),
        )

        with pytest.raises(HandlerError, match="not found"):
            async for _ in handler.handle("test"):
                pass


# ---------------------------------------------------------------------------
# AgentHandler — stop checks
# ---------------------------------------------------------------------------


class TestStopChecks:
    def test_workflow_stop_last_agent(self) -> None:
        """Workflow stops after the last agent in flow_order."""
        handler = AgentHandler(
            agents={"a": Agent(name="a"), "b": Agent(name="b")},
            flow_order=["a", "b"],
        )

        assert handler._check_workflow_stop("b") is True
        assert handler._check_workflow_stop("a") is False

    def test_workflow_stop_empty_flow(self) -> None:
        """Workflow stop returns True for empty flow_order."""
        handler = AgentHandler(agents={}, flow_order=[])

        assert handler._check_workflow_stop("any") is True

    def test_handoff_stop_no_handoff(self) -> None:
        """Handoff stops when no handoff target is detected."""
        agent = Agent(name="bot")
        handler = AgentHandler(agents={"bot": agent})
        result = RunResult(output="Final answer")

        assert handler._check_handoff_stop(result, agent) is True

    def test_handoff_stop_with_handoff(self) -> None:
        """Handoff continues when handoff target is detected."""
        target = Agent(name="target")
        agent = Agent(name="bot", handoffs=[target])
        handler = AgentHandler(agents={"bot": agent, "target": target})
        result = RunResult(output="target")

        assert handler._check_handoff_stop(result, agent) is False

    def test_team_stop_after_lead(self) -> None:
        """Team stops after the lead agent (first in flow_order)."""
        handler = AgentHandler(
            agents={"lead": Agent(name="lead"), "worker": Agent(name="worker")},
            flow_order=["lead", "worker"],
        )

        assert handler._check_team_stop("lead") is True
        assert handler._check_team_stop("worker") is False


# ---------------------------------------------------------------------------
# AgentHandler — handoff detection
# ---------------------------------------------------------------------------


class TestHandoffDetection:
    def test_detect_handoff_match(self) -> None:
        """Handoff detected when output exactly matches a target name."""
        target = Agent(name="billing")
        agent = Agent(name="triage", handoffs=[target])
        handler = AgentHandler(agents={"triage": agent, "billing": target})
        result = RunResult(output="billing")

        assert handler._detect_handoff(agent, result) == "billing"

    def test_detect_handoff_no_match(self) -> None:
        """No handoff when output doesn't match any target."""
        target = Agent(name="billing")
        agent = Agent(name="triage", handoffs=[target])
        handler = AgentHandler(agents={"triage": agent, "billing": target})
        result = RunResult(output="I can help you directly")

        assert handler._detect_handoff(agent, result) is None

    def test_detect_handoff_no_handoffs(self) -> None:
        """No handoff when agent has no handoff targets."""
        agent = Agent(name="bot")
        handler = AgentHandler(agents={"bot": agent})
        result = RunResult(output="anything")

        assert handler._detect_handoff(agent, result) is None

    def test_detect_handoff_target_not_in_swarm(self) -> None:
        """No handoff when target agent is not registered in the swarm."""
        target = Agent(name="external")
        agent = Agent(name="bot", handoffs=[target])
        # "external" is a handoff target but NOT in the handler's agents dict
        handler = AgentHandler(agents={"bot": agent})
        result = RunResult(output="external")

        assert handler._detect_handoff(agent, result) is None

    def test_detect_handoff_whitespace_stripped(self) -> None:
        """Handoff detection strips whitespace from output."""
        target = Agent(name="billing")
        agent = Agent(name="triage", handoffs=[target])
        handler = AgentHandler(agents={"triage": agent, "billing": target})
        result = RunResult(output="  billing  ")

        assert handler._detect_handoff(agent, result) == "billing"
