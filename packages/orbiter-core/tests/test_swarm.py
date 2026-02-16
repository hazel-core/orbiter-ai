"""Tests for orbiter.swarm — Swarm multi-agent orchestration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from orbiter.agent import Agent
from orbiter.runner import run
from orbiter.swarm import Swarm, SwarmError, _DelegateTool
from orbiter.types import AgentOutput, RunResult, ToolCall, Usage

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
# Swarm construction
# ---------------------------------------------------------------------------


class TestSwarmConstruction:
    def test_minimal_swarm(self) -> None:
        """Swarm can be created with a single agent and no flow DSL."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a])

        assert swarm.mode == "workflow"
        assert swarm.flow_order == ["a"]
        assert "a" in swarm.agents

    def test_swarm_with_flow_dsl(self) -> None:
        """Swarm parses flow DSL and determines topological order."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")

        assert swarm.flow_order == ["a", "b", "c"]
        assert swarm.flow == "a >> b >> c"

    def test_swarm_flow_order_from_dsl(self) -> None:
        """Flow DSL determines order independent of agent list order."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        # Agents given in reverse, but flow specifies a >> b >> c
        swarm = Swarm(agents=[c, b, a], flow="a >> b >> c")

        assert swarm.flow_order == ["a", "b", "c"]

    def test_swarm_default_order_is_agent_list_order(self) -> None:
        """Without flow DSL, order matches agent list."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        swarm = Swarm(agents=[c, a, b])

        assert swarm.flow_order == ["c", "a", "b"]

    def test_swarm_empty_agents_raises(self) -> None:
        """Swarm with no agents raises SwarmError."""
        with pytest.raises(SwarmError, match="at least one agent"):
            Swarm(agents=[])

    def test_swarm_duplicate_agent_names(self) -> None:
        """Swarm with duplicate agent names raises SwarmError."""
        a1 = Agent(name="a")
        a2 = Agent(name="a")

        with pytest.raises(SwarmError, match="Duplicate agent name"):
            Swarm(agents=[a1, a2])

    def test_swarm_flow_references_unknown_agent(self) -> None:
        """Flow DSL referencing an agent not in the swarm raises SwarmError."""
        a = Agent(name="a")

        with pytest.raises(SwarmError, match="unknown agent 'z'"):
            Swarm(agents=[a], flow="a >> z")

    def test_swarm_invalid_flow_dsl(self) -> None:
        """Invalid flow DSL string raises SwarmError."""
        a = Agent(name="a")

        with pytest.raises(SwarmError, match="Invalid flow DSL"):
            Swarm(agents=[a], flow="")

    def test_swarm_describe(self) -> None:
        """Swarm.describe() returns mode, flow, and agent info."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")

        desc = swarm.describe()

        assert desc["mode"] == "workflow"
        assert desc["flow"] == "a >> b"
        assert desc["flow_order"] == ["a", "b"]
        assert "a" in desc["agents"]
        assert "b" in desc["agents"]

    def test_swarm_repr(self) -> None:
        """Swarm.__repr__() includes mode, agents, flow."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a], flow="a")

        r = repr(swarm)

        assert "workflow" in r
        assert "a" in r


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------


class TestSwarmWorkflow:
    async def test_single_agent_workflow(self) -> None:
        """Workflow with one agent returns its output."""
        a = Agent(name="a", instructions="Be agent A.")
        swarm = Swarm(agents=[a])
        provider = _make_provider([AgentOutput(text="Hello from A")])

        result = await swarm.run("Hi", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "Hello from A"

    async def test_two_agent_pipeline(self) -> None:
        """Workflow chains output→input between two agents."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")

        # Agent a receives "Hi" and outputs "from_a"
        # Agent b receives "from_a" and outputs "from_b"
        provider = _make_provider(
            [
                AgentOutput(text="from_a"),
                AgentOutput(text="from_b"),
            ]
        )

        result = await swarm.run("Hi", provider=provider)

        assert result.output == "from_b"

    async def test_three_agent_pipeline(self) -> None:
        """Workflow chains through 3 agents sequentially."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")

        provider = _make_provider(
            [
                AgentOutput(text="step1"),
                AgentOutput(text="step2"),
                AgentOutput(text="step3"),
            ]
        )

        result = await swarm.run("start", provider=provider)

        assert result.output == "step3"

    async def test_workflow_output_becomes_next_input(self) -> None:
        """Each agent receives previous agent's output as input."""
        received_inputs: list[str] = []

        # Track what each agent receives by intercepting provider
        call_count = 0

        async def tracked_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            # Extract the last user message content to see what the agent received
            for m in reversed(messages):
                content = getattr(m, "content", None)
                role = getattr(m, "role", None)
                if role == "user" and content:
                    received_inputs.append(content)
                    break

            responses = [
                AgentOutput(text="output_from_a"),
                AgentOutput(text="output_from_b"),
                AgentOutput(text="output_from_c"),
            ]
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = tracked_complete

        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")

        await swarm.run("initial_input", provider=provider)

        assert received_inputs[0] == "initial_input"
        assert received_inputs[1] == "output_from_a"
        assert received_inputs[2] == "output_from_b"


# ---------------------------------------------------------------------------
# Swarm via run() public API
# ---------------------------------------------------------------------------


class TestSwarmViaRun:
    async def test_run_with_swarm(self) -> None:
        """run(swarm, ...) detects Swarm and delegates correctly."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")

        provider = _make_provider(
            [
                AgentOutput(text="from_a"),
                AgentOutput(text="from_b"),
            ]
        )

        result = await run(swarm, "test", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "from_b"

    def test_run_sync_with_swarm(self) -> None:
        """run.sync(swarm, ...) works for synchronous execution."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a])

        provider = _make_provider([AgentOutput(text="sync_ok")])

        result = run.sync(swarm, "test", provider=provider)

        assert result.output == "sync_ok"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSwarmEdgeCases:
    async def test_workflow_with_usage_tracking(self) -> None:
        """Workflow returns usage from the final agent."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")

        provider = _make_provider(
            [
                AgentOutput(
                    text="from_a",
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                ),
                AgentOutput(
                    text="from_b",
                    usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
                ),
            ]
        )

        result = await swarm.run("test", provider=provider)

        assert result.output == "from_b"
        # Usage comes from the final call_runner result
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 10

    async def test_unsupported_mode_raises(self) -> None:
        """Unsupported swarm mode raises SwarmError."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a], mode="invalid")

        provider = _make_provider([AgentOutput(text="ok")])

        with pytest.raises(SwarmError, match="Unsupported swarm mode"):
            await swarm.run("test", provider=provider)

    async def test_workflow_messages_passed_through(self) -> None:
        """Prior messages are forwarded to agents in workflow."""
        from orbiter.types import UserMessage

        a = Agent(name="a")
        swarm = Swarm(agents=[a])
        provider = _make_provider([AgentOutput(text="continued")])

        prior = [UserMessage(content="Earlier context")]
        result = await swarm.run("Continue", messages=prior, provider=provider)

        assert result.output == "continued"

    async def test_swarm_name_attribute(self) -> None:
        """Swarm has a name attribute for compatibility."""
        a = Agent(name="leader")
        swarm = Swarm(agents=[a])

        assert "leader" in swarm.name


# ---------------------------------------------------------------------------
# Handoff mode — construction
# ---------------------------------------------------------------------------


class TestSwarmHandoffConstruction:
    def test_handoff_mode_creation(self) -> None:
        """Swarm can be created in handoff mode."""
        triage = Agent(name="triage")
        billing = Agent(name="billing")
        swarm = Swarm(agents=[triage, billing], mode="handoff")

        assert swarm.mode == "handoff"
        assert swarm.max_handoffs == 10

    def test_handoff_mode_custom_max_handoffs(self) -> None:
        """max_handoffs can be configured."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], mode="handoff", max_handoffs=3)

        assert swarm.max_handoffs == 3

    def test_handoff_mode_no_flow_required(self) -> None:
        """Handoff mode works without flow DSL."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], mode="handoff")

        assert swarm.flow is None
        assert swarm.flow_order == ["a", "b"]


# ---------------------------------------------------------------------------
# Handoff mode — execution
# ---------------------------------------------------------------------------


class TestSwarmHandoff:
    async def test_simple_handoff_a_to_b(self) -> None:
        """Agent A hands off to agent B, B produces final output."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")

        # Agent a outputs "b" (matching handoff target name) -> triggers handoff
        # Agent b outputs "final answer" (no handoff) -> stops
        provider = _make_provider(
            [
                AgentOutput(text="b"),
                AgentOutput(text="final answer"),
            ]
        )

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "final answer"

    async def test_no_handoff_returns_immediately(self) -> None:
        """Agent with no handoff targets returns its output directly."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], mode="handoff")

        provider = _make_provider([AgentOutput(text="no handoff here")])

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "no handoff here"

    async def test_handoff_chain_a_to_b_to_c(self) -> None:
        """Handoff chain: A -> B -> C, C produces final output."""
        c = Agent(name="c")
        b = Agent(name="b", handoffs=[c])
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b, c], mode="handoff")

        provider = _make_provider(
            [
                AgentOutput(text="b"),  # a -> handoff to b
                AgentOutput(text="c"),  # b -> handoff to c
                AgentOutput(text="done!"),  # c -> final
            ]
        )

        result = await swarm.run("start", provider=provider)

        assert result.output == "done!"

    async def test_handoff_output_not_matching_target(self) -> None:
        """Agent with handoffs but output doesn't match any target name."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")

        # Agent a outputs something that is NOT "b" -> no handoff
        provider = _make_provider([AgentOutput(text="some regular answer")])

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "some regular answer"

    async def test_handoff_with_whitespace_stripping(self) -> None:
        """Handoff detection strips whitespace from output."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")

        # Output has leading/trailing whitespace around target name
        provider = _make_provider(
            [
                AgentOutput(text="  b  "),
                AgentOutput(text="handled by b"),
            ]
        )

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "handled by b"

    async def test_handoff_target_not_in_swarm(self) -> None:
        """Handoff target must exist in the swarm's agents dict."""
        external = Agent(name="external")
        a = Agent(name="a", handoffs=[external])

        # external is a handoff target on agent a, but NOT in the swarm
        swarm = Swarm(agents=[a], mode="handoff")

        # Agent a outputs "external" which matches handoff target name
        # but "external" is not in swarm.agents, so no handoff
        provider = _make_provider([AgentOutput(text="external")])

        result = await swarm.run("Hello", provider=provider)

        assert result.output == "external"

    async def test_handoff_conversation_history_transferred(self) -> None:
        """Handoff transfers conversation history to the next agent."""
        received_messages: list[list[Any]] = []

        call_count = 0

        async def tracked_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            received_messages.append(list(messages))

            responses = [
                AgentOutput(text="b"),  # a -> handoff to b
                AgentOutput(text="final"),  # b -> done
            ]
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = tracked_complete

        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        swarm = Swarm(agents=[a, b], mode="handoff")

        result = await swarm.run("Hello", provider=provider)

        # Handoff happened: 2 LLM calls (one per agent)
        assert len(received_messages) == 2
        # Agent b receives the handoff output ("b") as its input
        last_user_msg = None
        for m in received_messages[1]:
            if getattr(m, "role", None) == "user":
                last_user_msg = m
        assert last_user_msg is not None
        assert last_user_msg.content == "b"
        # Final result comes from agent b
        assert result.output == "final"


# ---------------------------------------------------------------------------
# Handoff mode — loop detection
# ---------------------------------------------------------------------------


class TestSwarmHandoffLoopDetection:
    async def test_loop_detection_triggers(self) -> None:
        """Endless handoff loop raises SwarmError."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        # Make b hand off back to a
        b.handoffs = {"a": a}

        swarm = Swarm(agents=[a, b], mode="handoff", max_handoffs=3)

        # a -> b -> a -> b -> exceeds max_handoffs=3
        provider = _make_provider(
            [
                AgentOutput(text="b"),  # a -> b
                AgentOutput(text="a"),  # b -> a
                AgentOutput(text="b"),  # a -> b
                AgentOutput(text="a"),  # b -> a (exceeds limit)
            ]
        )

        with pytest.raises(SwarmError, match=r"Max handoffs.*3.*exceeded"):
            await swarm.run("Hello", provider=provider)

    async def test_loop_detection_max_handoffs_1(self) -> None:
        """max_handoffs=1 allows exactly one handoff."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff", max_handoffs=1)

        provider = _make_provider(
            [
                AgentOutput(text="b"),  # a -> b (1 handoff, allowed)
                AgentOutput(text="result"),  # b -> final
            ]
        )

        result = await swarm.run("Hello", provider=provider)
        assert result.output == "result"

    async def test_loop_detection_max_handoffs_exceeded_exactly(self) -> None:
        """max_handoffs=1 fails on second handoff attempt."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        b.handoffs = {"a": a}

        swarm = Swarm(agents=[a, b], mode="handoff", max_handoffs=1)

        provider = _make_provider(
            [
                AgentOutput(text="b"),  # a -> b (1 handoff, OK)
                AgentOutput(text="a"),  # b -> a (2nd handoff, exceeds 1)
            ]
        )

        with pytest.raises(SwarmError, match=r"Max handoffs.*1.*exceeded"):
            await swarm.run("Hello", provider=provider)


# ---------------------------------------------------------------------------
# Handoff via run() public API
# ---------------------------------------------------------------------------


class TestSwarmHandoffViaRun:
    async def test_run_handoff_swarm(self) -> None:
        """run(handoff_swarm, ...) works correctly."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")

        provider = _make_provider(
            [
                AgentOutput(text="b"),
                AgentOutput(text="via run()"),
            ]
        )

        result = await run(swarm, "test", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "via run()"

    def test_run_sync_handoff_swarm(self) -> None:
        """run.sync(handoff_swarm, ...) works for synchronous execution."""
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])

        swarm = Swarm(agents=[a, b], mode="handoff")

        provider = _make_provider(
            [
                AgentOutput(text="b"),
                AgentOutput(text="sync handoff"),
            ]
        )

        result = run.sync(swarm, "test", provider=provider)

        assert result.output == "sync handoff"


# ---------------------------------------------------------------------------
# Team mode — construction
# ---------------------------------------------------------------------------


class TestSwarmTeamConstruction:
    def test_team_mode_creation(self) -> None:
        """Swarm can be created in team mode."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        assert swarm.mode == "team"
        assert swarm.flow_order == ["lead", "worker"]

    async def test_team_mode_single_agent_raises(self) -> None:
        """Team mode with only one agent raises SwarmError."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a], mode="team")

        provider = _make_provider([AgentOutput(text="ok")])

        with pytest.raises(SwarmError, match="at least two agents"):
            await swarm.run("test", provider=provider)


# ---------------------------------------------------------------------------
# Team mode — execution
# ---------------------------------------------------------------------------


_DEFAULT_USAGE = Usage(input_tokens=10, output_tokens=5, total_tokens=15)


def _make_team_provider(responses: list[Any]) -> Any:
    """Create a mock provider for team mode tests.

    Responses can be AgentOutput (text-only) or dicts with 'tool_calls'.
    Each response is consumed in order across all agents.
    """
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        if isinstance(resp, AgentOutput):

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        # Dict form for tool call responses
        class FakeToolCallResponse:
            content = resp.get("content", "")
            tool_calls = resp.get("tool_calls", [])
            usage = resp.get("usage", _DEFAULT_USAGE)

        return FakeToolCallResponse()

    mock = AsyncMock()
    mock.complete = complete
    return mock


class TestSwarmTeam:
    async def test_lead_delegates_to_worker(self) -> None:
        """Lead agent delegates to worker, then synthesizes result."""
        lead = Agent(name="lead", instructions="You are the lead.")
        worker = Agent(name="worker", instructions="You are a worker.")
        swarm = Swarm(agents=[lead, worker], mode="team")

        # Response 1 (lead): call delegate_to_worker tool
        tc = ToolCall(
            id="tc-1",
            name="delegate_to_worker",
            arguments=json.dumps({"task": "research topic X"}),
        )
        # Response 2 (worker): worker's output
        # Response 3 (lead): lead synthesizes final answer
        provider = _make_team_provider(
            [
                {"content": "", "tool_calls": [tc]},
                AgentOutput(text="worker found: topic X details"),
                AgentOutput(text="Based on research: final synthesis"),
            ]
        )

        result = await swarm.run("Summarize topic X", provider=provider)

        assert result.output == "Based on research: final synthesis"

    async def test_lead_delegates_to_multiple_workers(self) -> None:
        """Lead can delegate to different workers."""
        lead = Agent(name="lead")
        researcher = Agent(name="researcher")
        writer = Agent(name="writer")
        swarm = Swarm(agents=[lead, researcher, writer], mode="team")

        # Lead calls delegate_to_researcher, then delegate_to_writer, then returns
        tc1 = ToolCall(
            id="tc-1",
            name="delegate_to_researcher",
            arguments=json.dumps({"task": "research"}),
        )
        tc2 = ToolCall(
            id="tc-2",
            name="delegate_to_writer",
            arguments=json.dumps({"task": "write report"}),
        )
        provider = _make_team_provider(
            [
                {"content": "", "tool_calls": [tc1]},  # lead calls researcher
                AgentOutput(text="research results"),  # researcher responds
                {"content": "", "tool_calls": [tc2]},  # lead calls writer
                AgentOutput(text="written report"),  # writer responds
                AgentOutput(text="All done: final output"),  # lead final
            ]
        )

        result = await swarm.run("Do research and write", provider=provider)

        assert result.output == "All done: final output"

    async def test_lead_no_delegation_returns_directly(self) -> None:
        """Lead can return without delegating to any worker."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        # Lead responds directly without calling any delegate tool
        provider = _make_team_provider([AgentOutput(text="I can handle this myself")])

        result = await swarm.run("Simple question", provider=provider)

        assert result.output == "I can handle this myself"

    async def test_worker_result_visible_to_lead(self) -> None:
        """Worker output is returned to lead as tool result."""
        received_messages: list[list[Any]] = []
        call_count = 0

        async def tracked_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            received_messages.append(list(messages))

            tc = ToolCall(
                id="tc-1",
                name="delegate_to_worker",
                arguments=json.dumps({"task": "do work"}),
            )
            responses: list[Any] = [
                {"content": "", "tool_calls": [tc]},  # lead delegates
                AgentOutput(text="worker result 42"),  # worker responds
                AgentOutput(text="synthesized"),  # lead final
            ]
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1

            if isinstance(resp, AgentOutput):

                class FR:
                    content = resp.text
                    tool_calls = resp.tool_calls
                    usage = resp.usage

                return FR()

            class FTR:
                content = resp.get("content", "")
                tool_calls = resp.get("tool_calls", [])
                usage = resp.get("usage", _DEFAULT_USAGE)

            return FTR()

        provider = AsyncMock()
        provider.complete = tracked_complete

        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        result = await swarm.run("test", provider=provider)

        assert result.output == "synthesized"
        # At least 3 calls: lead (delegate), worker, lead (synthesize)
        assert len(received_messages) >= 3
        # The third call (lead's second) should contain the tool result
        # with the worker's output
        lead_second_call_msgs = received_messages[2]
        tool_results = [m for m in lead_second_call_msgs if getattr(m, "role", None) == "tool"]
        assert len(tool_results) > 0
        assert "worker result 42" in tool_results[0].content

    async def test_team_tools_restored_after_run(self) -> None:
        """Lead's original tools are restored after team execution."""
        from orbiter.tool import FunctionTool

        def my_tool(x: str) -> str:
            """A custom tool."""
            return x

        original_tool = FunctionTool(my_tool, name="my_tool")
        lead = Agent(name="lead", tools=[original_tool])
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        provider = _make_team_provider([AgentOutput(text="no delegation")])

        await swarm.run("test", provider=provider)

        # Lead should have only its original tool, not delegate tools
        assert "my_tool" in lead.tools
        assert "delegate_to_worker" not in lead.tools
        assert len(lead.tools) == 1

    async def test_team_tools_restored_on_error(self) -> None:
        """Lead's tools are restored even if run() raises."""
        from orbiter._internal.call_runner import CallRunnerError

        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        # Provider that always raises
        async def failing_complete(messages: Any, **kwargs: Any) -> Any:
            raise RuntimeError("LLM down")

        provider = AsyncMock()
        provider.complete = failing_complete

        with pytest.raises(CallRunnerError):
            await swarm.run("test", provider=provider)

        # Delegate tools should NOT remain on the lead
        assert "delegate_to_worker" not in lead.tools


# ---------------------------------------------------------------------------
# Team mode — via run() public API
# ---------------------------------------------------------------------------


class TestSwarmTeamViaRun:
    async def test_run_team_swarm(self) -> None:
        """run(team_swarm, ...) works correctly."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        tc = ToolCall(
            id="tc-1",
            name="delegate_to_worker",
            arguments=json.dumps({"task": "work"}),
        )
        provider = _make_team_provider(
            [
                {"content": "", "tool_calls": [tc]},
                AgentOutput(text="worker done"),
                AgentOutput(text="via run()"),
            ]
        )

        result = await run(swarm, "test", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "via run()"

    def test_run_sync_team_swarm(self) -> None:
        """run.sync(team_swarm, ...) works for synchronous execution."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        provider = _make_team_provider([AgentOutput(text="sync team")])

        result = run.sync(swarm, "test", provider=provider)

        assert result.output == "sync team"


# ---------------------------------------------------------------------------
# _DelegateTool unit tests
# ---------------------------------------------------------------------------


class TestDelegateTool:
    def test_delegate_tool_name(self) -> None:
        """DelegateTool has correct name and schema."""
        worker = Agent(name="researcher")
        dtool = _DelegateTool(worker=worker)

        assert dtool.name == "delegate_to_researcher"
        assert "task" in dtool.parameters["properties"]
        assert dtool.parameters["required"] == ["task"]

    def test_delegate_tool_schema(self) -> None:
        """DelegateTool generates valid OpenAI function-calling schema."""
        worker = Agent(name="writer")
        dtool = _DelegateTool(worker=worker)

        schema = dtool.to_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "delegate_to_writer"
        assert "writer" in schema["function"]["description"]

    async def test_delegate_tool_execute(self) -> None:
        """DelegateTool executes the worker agent and returns output."""
        worker = Agent(name="worker")
        provider = _make_provider([AgentOutput(text="worker output")])
        dtool = _DelegateTool(worker=worker, provider=provider)

        result = await dtool.execute(task="do something")

        assert result == "worker output"
