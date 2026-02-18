"""Tests for Agent and Swarm serialization/deserialization (US-015)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from orbiter.agent import Agent, _deserialize_tool, _import_object, _serialize_tool
from orbiter.swarm import Swarm
from orbiter.tool import FunctionTool, Tool, tool

# ---------------------------------------------------------------------------
# Module-level fixtures (importable for serialization round-trips)
# ---------------------------------------------------------------------------


@tool
def search(query: str) -> str:
    """Search for something."""
    return f"Results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return f"Result: {expression}"


class SummaryOutput(BaseModel):
    title: str
    body: str


# A custom Tool subclass at module level
class CustomSearchTool(Tool):
    def __init__(self) -> None:
        self.name = "custom_search"
        self.description = "Custom search tool"
        self.parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "custom result"


# ---------------------------------------------------------------------------
# Agent.to_dict() tests
# ---------------------------------------------------------------------------


class TestAgentToDict:
    def test_minimal_agent(self) -> None:
        """Minimal agent serializes with default values."""
        agent = Agent(name="bot", model="openai:gpt-4o")
        data = agent.to_dict()
        assert data["name"] == "bot"
        assert data["model"] == "openai:gpt-4o"
        assert data["instructions"] == ""
        assert data["max_steps"] == 10
        assert data["temperature"] == 1.0
        assert data["max_tokens"] is None
        assert "tools" not in data
        assert "handoffs" not in data
        assert "output_type" not in data

    def test_with_tools(self) -> None:
        """Agent with tools serializes tool paths."""
        agent = Agent(name="bot", tools=[search, calculate])
        data = agent.to_dict()
        assert "tools" in data
        assert len(data["tools"]) == 2
        # Tools should be importable dotted paths
        for path in data["tools"]:
            assert isinstance(path, str)
            assert "." in path

    def test_with_handoffs(self) -> None:
        """Agent with handoffs serializes them recursively."""
        target = Agent(name="target", model="openai:gpt-4o-mini")
        agent = Agent(name="router", handoffs=[target])
        data = agent.to_dict()
        assert "handoffs" in data
        assert len(data["handoffs"]) == 1
        assert data["handoffs"][0]["name"] == "target"
        assert data["handoffs"][0]["model"] == "openai:gpt-4o-mini"

    def test_with_output_type(self) -> None:
        """Agent with output_type serializes it as importable path."""
        agent = Agent(name="bot", output_type=SummaryOutput)
        data = agent.to_dict()
        assert "output_type" in data
        assert "SummaryOutput" in data["output_type"]

    def test_with_custom_tool_subclass(self) -> None:
        """Agent with a custom Tool subclass serializes its class path."""
        custom = CustomSearchTool()
        agent = Agent(name="bot", tools=[custom])
        data = agent.to_dict()
        assert len(data["tools"]) == 1
        assert "CustomSearchTool" in data["tools"][0]

    def test_callable_instructions_raises(self) -> None:
        """Callable instructions cannot be serialized."""
        agent = Agent(name="bot", instructions=lambda name: "You are helpful")
        with pytest.raises(ValueError, match="callable instructions"):
            agent.to_dict()

    def test_hooks_raises(self) -> None:
        """Agent with hooks cannot be serialized."""

        async def my_hook(**kwargs: Any) -> None:
            pass

        from orbiter.hooks import HookPoint

        agent = Agent(name="bot", hooks=[(HookPoint.START, my_hook)])
        with pytest.raises(ValueError, match="hooks"):
            agent.to_dict()

    def test_memory_raises(self) -> None:
        """Agent with memory cannot be serialized."""
        agent = Agent(name="bot", memory="some_memory")
        with pytest.raises(ValueError, match="memory"):
            agent.to_dict()

    def test_context_raises(self) -> None:
        """Agent with context cannot be serialized."""
        agent = Agent(name="bot", context="some_context")
        with pytest.raises(ValueError, match="context"):
            agent.to_dict()

    def test_closure_tool_raises(self) -> None:
        """Closure-based tools cannot be serialized."""

        def make_tool() -> FunctionTool:
            x = 42

            def my_closure(query: str) -> str:
                return f"{x}: {query}"

            return FunctionTool(my_closure)

        agent = Agent(name="bot", tools=[make_tool()])
        with pytest.raises(ValueError, match="closure or lambda"):
            agent.to_dict()

    def test_json_serializable(self) -> None:
        """to_dict() output is JSON-serializable."""
        agent = Agent(name="bot", tools=[search], instructions="Be helpful")
        data = agent.to_dict()
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["name"] == "bot"


# ---------------------------------------------------------------------------
# Agent.from_dict() tests
# ---------------------------------------------------------------------------


class TestAgentFromDict:
    def test_minimal_reconstruction(self) -> None:
        """from_dict() reconstructs a minimal agent."""
        data = {"name": "bot", "model": "openai:gpt-4o"}
        agent = Agent.from_dict(data)
        assert agent.name == "bot"
        assert agent.model == "openai:gpt-4o"
        assert agent.max_steps == 10

    def test_with_all_fields(self) -> None:
        """from_dict() reconstructs agent with all scalar fields."""
        data = {
            "name": "assistant",
            "model": "anthropic:claude-3",
            "instructions": "You are a helpful assistant",
            "max_steps": 5,
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        agent = Agent.from_dict(data)
        assert agent.name == "assistant"
        assert agent.model == "anthropic:claude-3"
        assert agent.instructions == "You are a helpful assistant"
        assert agent.max_steps == 5
        assert agent.temperature == 0.7
        assert agent.max_tokens == 1000

    def test_with_tools_from_path(self) -> None:
        """from_dict() resolves tool paths to actual tools."""
        path = f"{search._fn.__module__}.{search._fn.__qualname__}"
        data = {"name": "bot", "tools": [path]}
        agent = Agent.from_dict(data)
        assert len(agent.tools) == 1
        assert "search" in agent.tools

    def test_with_handoffs(self) -> None:
        """from_dict() reconstructs handoff agents recursively."""
        data = {
            "name": "router",
            "handoffs": [{"name": "target", "model": "openai:gpt-4o-mini"}],
        }
        agent = Agent.from_dict(data)
        assert len(agent.handoffs) == 1
        assert "target" in agent.handoffs
        assert agent.handoffs["target"].model == "openai:gpt-4o-mini"

    def test_with_output_type(self) -> None:
        """from_dict() resolves output_type from importable path."""
        path = f"{SummaryOutput.__module__}.{SummaryOutput.__qualname__}"
        data = {"name": "bot", "output_type": path}
        agent = Agent.from_dict(data)
        assert agent.output_type is SummaryOutput

    def test_invalid_tool_path_raises(self) -> None:
        """from_dict() raises on unresolvable tool path."""
        data = {"name": "bot", "tools": ["nonexistent.module.func"]}
        with pytest.raises(ValueError, match="Cannot import"):
            Agent.from_dict(data)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestAgentRoundTrip:
    def test_minimal_round_trip(self) -> None:
        """Minimal agent survives to_dict/from_dict round-trip."""
        original = Agent(name="bot", model="openai:gpt-4o")
        reconstructed = Agent.from_dict(original.to_dict())
        assert reconstructed.name == original.name
        assert reconstructed.model == original.model
        assert reconstructed.max_steps == original.max_steps

    def test_full_round_trip(self) -> None:
        """Agent with tools, handoffs, and output_type round-trips correctly."""
        target = Agent(name="helper", model="openai:gpt-4o-mini")
        original = Agent(
            name="main",
            model="anthropic:claude-3",
            instructions="Be helpful",
            tools=[search, calculate],
            handoffs=[target],
            output_type=SummaryOutput,
            max_steps=5,
            temperature=0.7,
            max_tokens=2000,
        )
        data = original.to_dict()
        reconstructed = Agent.from_dict(data)

        assert reconstructed.name == original.name
        assert reconstructed.model == original.model
        assert reconstructed.instructions == original.instructions
        assert reconstructed.max_steps == original.max_steps
        assert reconstructed.temperature == original.temperature
        assert reconstructed.max_tokens == original.max_tokens
        assert set(reconstructed.tools.keys()) == set(original.tools.keys())
        assert set(reconstructed.handoffs.keys()) == set(original.handoffs.keys())
        assert reconstructed.output_type is original.output_type

    def test_json_round_trip(self) -> None:
        """Agent survives JSON serialization/deserialization round-trip."""
        original = Agent(
            name="bot",
            model="openai:gpt-4o",
            tools=[search],
            instructions="Be concise",
        )
        json_str = json.dumps(original.to_dict())
        data = json.loads(json_str)
        reconstructed = Agent.from_dict(data)
        assert reconstructed.name == original.name
        assert "search" in reconstructed.tools

    def test_custom_tool_round_trip(self) -> None:
        """Custom Tool subclass round-trips via class path."""
        custom = CustomSearchTool()
        original = Agent(name="bot", tools=[custom])
        data = original.to_dict()
        reconstructed = Agent.from_dict(data)
        assert "custom_search" in reconstructed.tools


# ---------------------------------------------------------------------------
# Swarm serialization tests
# ---------------------------------------------------------------------------


class TestSwarmToDict:
    def test_basic_swarm(self) -> None:
        """Basic swarm serializes agents, flow, and mode."""
        a = Agent(name="a", model="openai:gpt-4o")
        b = Agent(name="b", model="openai:gpt-4o")
        swarm = Swarm(agents=[a, b], flow="a >> b")
        data = swarm.to_dict()
        assert data["mode"] == "workflow"
        assert data["flow"] == "a >> b"
        assert data["max_handoffs"] == 10
        assert len(data["agents"]) == 2

    def test_handoff_mode(self) -> None:
        """Handoff mode swarm serializes correctly."""
        a = Agent(name="a", handoffs=[Agent(name="b")])
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b", mode="handoff", max_handoffs=5)
        data = swarm.to_dict()
        assert data["mode"] == "handoff"
        assert data["max_handoffs"] == 5

    def test_no_flow(self) -> None:
        """Swarm without explicit flow serializes flow as None."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b])
        data = swarm.to_dict()
        assert data["flow"] is None


class TestSwarmFromDict:
    def test_basic_reconstruction(self) -> None:
        """from_dict() reconstructs a basic swarm."""
        data = {
            "agents": [
                {"name": "a", "model": "openai:gpt-4o"},
                {"name": "b", "model": "openai:gpt-4o"},
            ],
            "flow": "a >> b",
            "mode": "workflow",
            "max_handoffs": 10,
        }
        swarm = Swarm.from_dict(data)
        assert swarm.mode == "workflow"
        assert swarm.flow == "a >> b"
        assert len(swarm.agents) == 2
        assert "a" in swarm.agents
        assert "b" in swarm.agents

    def test_no_flow_reconstruction(self) -> None:
        """from_dict() handles None flow (uses agent list order)."""
        data = {
            "agents": [
                {"name": "x"},
                {"name": "y"},
            ],
            "flow": None,
            "mode": "workflow",
        }
        swarm = Swarm.from_dict(data)
        assert swarm.flow_order == ["x", "y"]


class TestSwarmRoundTrip:
    def test_workflow_round_trip(self) -> None:
        """Workflow swarm round-trips correctly."""
        a = Agent(name="a", model="openai:gpt-4o")
        b = Agent(name="b", model="openai:gpt-4o-mini")
        original = Swarm(agents=[a, b], flow="a >> b")
        reconstructed = Swarm.from_dict(original.to_dict())
        assert reconstructed.mode == original.mode
        assert reconstructed.flow == original.flow
        assert reconstructed.flow_order == original.flow_order
        assert set(reconstructed.agents.keys()) == set(original.agents.keys())

    def test_json_round_trip(self) -> None:
        """Swarm survives JSON round-trip."""
        a = Agent(name="a", tools=[search])
        b = Agent(name="b")
        original = Swarm(agents=[a, b], flow="a >> b", mode="workflow")
        json_str = json.dumps(original.to_dict())
        data = json.loads(json_str)
        reconstructed = Swarm.from_dict(data)
        assert reconstructed.mode == "workflow"
        assert "search" in reconstructed.agents["a"].tools

    def test_team_mode_round_trip(self) -> None:
        """Team mode swarm round-trips correctly."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        original = Swarm(agents=[lead, worker], mode="team")
        reconstructed = Swarm.from_dict(original.to_dict())
        assert reconstructed.mode == "team"
        assert "lead" in reconstructed.agents
        assert "worker" in reconstructed.agents


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestSerializationHelpers:
    def test_serialize_function_tool(self) -> None:
        """FunctionTool serializes to its wrapped function's dotted path."""
        path = _serialize_tool(search)
        assert "search" in path
        assert "." in path

    def test_serialize_custom_tool(self) -> None:
        """Custom Tool subclass serializes to its class dotted path."""
        custom = CustomSearchTool()
        path = _serialize_tool(custom)
        assert "CustomSearchTool" in path

    def test_deserialize_function_tool(self) -> None:
        """Deserializing a FunctionTool path returns a Tool."""
        path = _serialize_tool(search)
        result = _deserialize_tool(path)
        assert isinstance(result, Tool)

    def test_deserialize_custom_tool_class(self) -> None:
        """Deserializing a custom Tool subclass path returns an instance."""
        custom = CustomSearchTool()
        path = _serialize_tool(custom)
        result = _deserialize_tool(path)
        assert isinstance(result, Tool)
        assert result.name == "custom_search"

    def test_import_object_valid(self) -> None:
        """_import_object resolves a valid dotted path."""
        obj = _import_object("orbiter.agent.Agent")
        assert obj is Agent

    def test_import_object_invalid(self) -> None:
        """_import_object raises on invalid path."""
        with pytest.raises(ValueError, match="Cannot import"):
            _import_object("nonexistent.module.Thing")

    def test_lambda_tool_raises(self) -> None:
        """Lambda tools cannot be serialized."""
        lam = FunctionTool(lambda x: x, name="lam")
        with pytest.raises(ValueError, match="closure or lambda"):
            _serialize_tool(lam)

    def test_deserialize_unknown_dict_raises(self) -> None:
        """Deserializing an unknown dict format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool dict format"):
            _deserialize_tool({"unknown": True})


# ---------------------------------------------------------------------------
# MCPToolWrapper serialization tests
# ---------------------------------------------------------------------------


class TestMCPToolWrapperSerialization:
    def test_serialize_mcp_tool_returns_dict(self) -> None:
        """MCPToolWrapper serializes to a dict with __mcp_tool__ marker."""
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from orbiter.mcp.tools import MCPToolWrapper

        mcp_tool = MCPTool(
            name="search",
            description="Search the web",
            inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        wrapper = MCPToolWrapper(mcp_tool, "my_server", AsyncMock())
        result = _serialize_tool(wrapper)
        assert isinstance(result, dict)
        assert result["__mcp_tool__"] is True
        assert result["original_name"] == "search"
        assert result["server_name"] == "my_server"

    def test_mcp_tool_round_trip(self) -> None:
        """MCPToolWrapper survives _serialize_tool / _deserialize_tool round-trip."""
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from orbiter.mcp.tools import MCPToolWrapper

        mcp_tool = MCPTool(
            name="read_file",
            description="Read a file",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        original = MCPToolWrapper(mcp_tool, "fs_server", AsyncMock())
        serialized = _serialize_tool(original)
        restored = _deserialize_tool(serialized)

        assert isinstance(restored, MCPToolWrapper)
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.parameters == original.parameters
        assert restored.original_name == original.original_name
        assert restored.server_name == original.server_name

    def test_mcp_tool_json_round_trip(self) -> None:
        """MCPToolWrapper survives JSON serialization round-trip."""
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from orbiter.mcp.client import MCPServerConfig
        from orbiter.mcp.tools import MCPToolWrapper

        config = MCPServerConfig(name="srv", transport="sse", url="http://localhost:8080")
        mcp_tool = MCPTool(
            name="query",
            description="Run a query",
            inputSchema={"type": "object", "properties": {}},
        )
        original = MCPToolWrapper(mcp_tool, "srv", AsyncMock(), server_config=config)
        serialized = _serialize_tool(original)
        json_str = json.dumps(serialized)
        data = json.loads(json_str)
        restored = _deserialize_tool(data)

        assert isinstance(restored, MCPToolWrapper)
        assert restored.name == original.name
        assert restored._server_config is not None
        assert restored._server_config.name == "srv"
        assert restored._server_config.url == "http://localhost:8080"

    def test_agent_with_mcp_tools_round_trip(self) -> None:
        """Agent with MCPToolWrapper tools survives to_dict/from_dict."""
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from orbiter.mcp.tools import MCPToolWrapper

        mcp_tool = MCPTool(
            name="search",
            description="Search",
            inputSchema={"type": "object", "properties": {}},
        )
        wrapper = MCPToolWrapper(mcp_tool, "srv", AsyncMock())
        agent = Agent(name="mcp-agent", tools=[wrapper])
        data = agent.to_dict()

        # Verify the tool is serialized as a dict
        assert isinstance(data["tools"][0], dict)
        assert data["tools"][0]["__mcp_tool__"] is True

        # Reconstruct
        restored = Agent.from_dict(data)
        assert len(restored.tools) == 1
        tool_name = list(restored.tools.keys())[0]
        assert "search" in tool_name

    def test_agent_with_mixed_tools_round_trip(self) -> None:
        """Agent with both regular and MCP tools round-trips correctly."""
        from unittest.mock import AsyncMock

        from mcp.types import Tool as MCPTool

        from orbiter.mcp.tools import MCPToolWrapper

        mcp_tool = MCPTool(
            name="mcp_search",
            description="MCP Search",
            inputSchema={"type": "object", "properties": {}},
        )
        mcp_wrapper = MCPToolWrapper(mcp_tool, "srv", AsyncMock())
        agent = Agent(name="mixed", tools=[search, mcp_wrapper])
        data = agent.to_dict()

        # One should be a string (FunctionTool), one a dict (MCPToolWrapper)
        types = {type(t) for t in data["tools"]}
        assert str in types
        assert dict in types

        restored = Agent.from_dict(data)
        assert len(restored.tools) == 2
