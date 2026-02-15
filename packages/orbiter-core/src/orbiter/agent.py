"""Agent class: the core autonomous unit in Orbiter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from orbiter.config import parse_model_string
from orbiter.hooks import Hook, HookManager, HookPoint
from orbiter.tool import Tool
from orbiter.types import OrbiterError


class AgentError(OrbiterError):
    """Raised for agent-level errors (duplicate tools, invalid config, etc.)."""


class Agent:
    """An autonomous LLM-powered agent with tools and lifecycle hooks.

    Agents are the core building block in Orbiter. Each agent wraps an LLM
    model, a set of tools, optional handoff targets, and lifecycle hooks.
    The ``run()`` method (added in a later session) executes the agent's
    tool loop.

    All parameters are keyword-only; only ``name`` is required.

    Args:
        name: Unique identifier for this agent.
        model: Model string in ``"provider:model_name"`` format.
        instructions: System prompt. Can be a string or an async callable
            that receives a context dict and returns a string.
        tools: Tools available to this agent.
        handoffs: Other agents this agent can delegate to via handoff.
        hooks: Lifecycle hooks as ``(HookPoint, Hook)`` tuples.
        output_type: Pydantic model class for structured output validation.
        max_steps: Maximum LLM-tool round-trips before stopping.
        temperature: LLM sampling temperature.
        max_tokens: Maximum output tokens per LLM call.
    """

    def __init__(
        self,
        *,
        name: str,
        model: str = "openai:gpt-4o",
        instructions: str | Callable[..., str] = "",
        tools: list[Tool] | None = None,
        handoffs: list[Agent] | None = None,
        hooks: list[tuple[HookPoint, Hook]] | None = None,
        output_type: type[BaseModel] | None = None,
        max_steps: int = 10,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.provider_name, self.model_name = parse_model_string(model)
        self.instructions = instructions
        self.output_type = output_type
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Tools indexed by name for O(1) lookup during execution
        self.tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self._register_tool(t)

        # Handoff targets indexed by name
        self.handoffs: dict[str, Agent] = {}
        if handoffs:
            for agent in handoffs:
                self._register_handoff(agent)

        # Lifecycle hooks
        self.hook_manager = HookManager()
        if hooks:
            for point, hook in hooks:
                self.hook_manager.add(point, hook)

    def _register_tool(self, t: Tool) -> None:
        """Add a tool, raising on duplicate names.

        Args:
            t: The tool to register.

        Raises:
            AgentError: If a tool with the same name is already registered.
        """
        if t.name in self.tools:
            raise AgentError(f"Duplicate tool name '{t.name}' on agent '{self.name}'")
        self.tools[t.name] = t

    def _register_handoff(self, agent: Agent) -> None:
        """Add a handoff target, raising on duplicate names.

        Args:
            agent: The target agent.

        Raises:
            AgentError: If a handoff with the same name is already registered.
        """
        if agent.name in self.handoffs:
            raise AgentError(f"Duplicate handoff agent '{agent.name}' on agent '{self.name}'")
        self.handoffs[agent.name] = agent

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool schemas for all registered tools.

        Returns:
            A list of tool schema dicts suitable for LLM function calling.
        """
        return [t.to_schema() for t in self.tools.values()]

    def describe(self) -> dict[str, Any]:
        """Return a summary of the agent's capabilities.

        Useful for debugging, logging, and capability advertisement
        in multi-agent systems.

        Returns:
            A dict with the agent's name, model, tools, and configuration.
        """
        return {
            "name": self.name,
            "model": self.model,
            "tools": list(self.tools.keys()),
            "handoffs": list(self.handoffs.keys()),
            "max_steps": self.max_steps,
            "output_type": (self.output_type.__name__ if self.output_type else None),
        }

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"model={self.model!r}"]
        if self.tools:
            parts.append(f"tools={list(self.tools.keys())}")
        if self.handoffs:
            parts.append(f"handoffs={list(self.handoffs.keys())}")
        return f"Agent({', '.join(parts)})"
