"""Tests for orbiter.agent — Agent class init & configuration."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from orbiter.agent import Agent, AgentError
from orbiter.hooks import HookPoint
from orbiter.tool import FunctionTool, Tool, tool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@tool
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class ReportOutput(BaseModel):
    title: str
    body: str


# ---------------------------------------------------------------------------
# Agent creation: minimal & full
# ---------------------------------------------------------------------------


class TestAgentCreation:
    def test_minimal(self) -> None:
        """Agent with only a name uses sensible defaults."""
        agent = Agent(name="bot")
        assert agent.name == "bot"
        assert agent.model == "openai:gpt-4o"
        assert agent.provider_name == "openai"
        assert agent.model_name == "gpt-4o"
        assert agent.instructions == ""
        assert agent.tools == {}
        assert agent.handoffs == {}
        assert agent.output_type is None
        assert agent.max_steps == 10
        assert agent.temperature == 1.0
        assert agent.max_tokens is None
        assert agent.memory is None
        assert agent.context is None

    def test_full_config(self) -> None:
        """Agent accepts all configuration parameters."""

        async def my_hook(**data: Any) -> None:
            pass

        agent = Agent(
            name="researcher",
            model="anthropic:claude-sonnet-4-20250514",
            instructions="Research things.",
            tools=[greet, add],
            hooks=[(HookPoint.PRE_LLM_CALL, my_hook)],
            output_type=ReportOutput,
            max_steps=20,
            temperature=0.7,
            max_tokens=4096,
        )
        assert agent.name == "researcher"
        assert agent.provider_name == "anthropic"
        assert agent.model_name == "claude-sonnet-4-20250514"
        assert agent.instructions == "Research things."
        assert len(agent.tools) == 2
        assert agent.output_type is ReportOutput
        assert agent.max_steps == 20
        assert agent.temperature == 0.7
        assert agent.max_tokens == 4096

    def test_name_is_required(self) -> None:
        """Agent() without name raises TypeError."""
        with pytest.raises(TypeError):
            Agent()  # type: ignore[call-arg]

    def test_all_params_keyword_only(self) -> None:
        """Positional arguments are not accepted."""
        with pytest.raises(TypeError):
            Agent("bot")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Model string parsing
# ---------------------------------------------------------------------------


class TestModelParsing:
    def test_provider_colon_model(self) -> None:
        agent = Agent(name="a", model="anthropic:claude-3-opus")
        assert agent.provider_name == "anthropic"
        assert agent.model_name == "claude-3-opus"

    def test_no_colon_defaults_openai(self) -> None:
        agent = Agent(name="a", model="gpt-4o-mini")
        assert agent.provider_name == "openai"
        assert agent.model_name == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_tools_stored_by_name(self) -> None:
        agent = Agent(name="bot", tools=[greet, add])
        assert "greet" in agent.tools
        assert "add" in agent.tools
        assert isinstance(agent.tools["greet"], FunctionTool)

    def test_duplicate_tool_raises(self) -> None:
        dup = FunctionTool(lambda x: x, name="greet")
        with pytest.raises(AgentError, match="Duplicate tool name 'greet'"):
            Agent(name="bot", tools=[greet, dup])

    def test_tool_abc_subclass(self) -> None:
        """Custom Tool subclasses are accepted."""

        class MyTool(Tool):
            name = "custom"
            description = "A custom tool."
            parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

            async def execute(self, **kwargs: Any) -> str:
                return "done"

        agent = Agent(name="bot", tools=[MyTool()])
        assert "custom" in agent.tools

    def test_get_tool_schemas(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        schemas = agent.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "greet"


# ---------------------------------------------------------------------------
# Handoff registration
# ---------------------------------------------------------------------------


class TestHandoffRegistration:
    def test_handoffs_stored_by_name(self) -> None:
        billing = Agent(name="billing")
        support = Agent(name="support")
        triage = Agent(name="triage", handoffs=[billing, support])
        assert "billing" in triage.handoffs
        assert "support" in triage.handoffs

    def test_duplicate_handoff_raises(self) -> None:
        a1 = Agent(name="helper")
        a2 = Agent(name="helper")
        with pytest.raises(AgentError, match="Duplicate handoff agent 'helper'"):
            Agent(name="main", handoffs=[a1, a2])


# ---------------------------------------------------------------------------
# Hook integration
# ---------------------------------------------------------------------------


class TestHookIntegration:
    def test_hook_manager_initialized(self) -> None:
        agent = Agent(name="bot")
        assert not agent.hook_manager.has_hooks(HookPoint.PRE_LLM_CALL)

    async def test_hooks_registered(self) -> None:
        calls: list[str] = []

        async def on_pre(**data: Any) -> None:
            calls.append("pre")

        agent = Agent(
            name="bot",
            hooks=[(HookPoint.PRE_LLM_CALL, on_pre)],
        )
        assert agent.hook_manager.has_hooks(HookPoint.PRE_LLM_CALL)
        await agent.hook_manager.run(HookPoint.PRE_LLM_CALL)
        assert calls == ["pre"]

    async def test_multiple_hooks(self) -> None:
        calls: list[str] = []

        async def hook_a(**data: Any) -> None:
            calls.append("a")

        async def hook_b(**data: Any) -> None:
            calls.append("b")

        agent = Agent(
            name="bot",
            hooks=[
                (HookPoint.START, hook_a),
                (HookPoint.FINISHED, hook_b),
            ],
        )
        await agent.hook_manager.run(HookPoint.START)
        await agent.hook_manager.run(HookPoint.FINISHED)
        assert calls == ["a", "b"]


# ---------------------------------------------------------------------------
# Instructions (str and callable)
# ---------------------------------------------------------------------------


class TestInstructions:
    def test_string_instructions(self) -> None:
        agent = Agent(name="bot", instructions="Be helpful.")
        assert agent.instructions == "Be helpful."

    def test_callable_instructions(self) -> None:
        def make_instructions(agent_name: str) -> str:
            return f"You are {agent_name}."

        agent = Agent(name="bot", instructions=make_instructions)
        assert callable(agent.instructions)
        assert agent.instructions("bot") == "You are bot."  # type: ignore[operator]


# ---------------------------------------------------------------------------
# describe() and __repr__
# ---------------------------------------------------------------------------


class TestDescribeAndRepr:
    def test_describe_minimal(self) -> None:
        agent = Agent(name="bot")
        desc = agent.describe()
        assert desc["name"] == "bot"
        assert desc["model"] == "openai:gpt-4o"
        assert desc["tools"] == []
        assert desc["handoffs"] == []
        assert desc["output_type"] is None

    def test_describe_with_tools_and_handoffs(self) -> None:
        helper = Agent(name="helper")
        agent = Agent(
            name="main",
            tools=[greet],
            handoffs=[helper],
            output_type=ReportOutput,
        )
        desc = agent.describe()
        assert desc["tools"] == ["greet"]
        assert desc["handoffs"] == ["helper"]
        assert desc["output_type"] == "ReportOutput"

    def test_repr_minimal(self) -> None:
        agent = Agent(name="bot")
        r = repr(agent)
        assert "Agent(" in r
        assert "name='bot'" in r
        assert "model='openai:gpt-4o'" in r

    def test_repr_with_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        r = repr(agent)
        assert "tools=['greet']" in r
