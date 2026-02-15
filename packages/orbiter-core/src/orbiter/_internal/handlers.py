"""Handler abstractions for composable agent execution.

Provides ``Handler[IN, OUT]`` as the base abstraction for processing
units that transform inputs to outputs via async generators, and
``AgentHandler`` for routing between agents in multi-agent swarms
with topology-aware stop conditions.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Generic, TypeVar

from orbiter._internal.call_runner import call_runner
from orbiter._internal.state import RunState
from orbiter.types import Message, OrbiterError, RunResult

IN = TypeVar("IN")
OUT = TypeVar("OUT")


class HandlerError(OrbiterError):
    """Raised for handler-level errors (routing, dispatch, stop checks)."""


class SwarmMode(StrEnum):
    """Swarm topology modes for agent orchestration."""

    WORKFLOW = "workflow"
    HANDOFF = "handoff"
    TEAM = "team"


class Handler(abc.ABC, Generic[IN, OUT]):
    """Abstract base for composable processing units.

    Handlers receive an input and yield zero or more outputs via
    an async generator.  This enables streaming, backpressure,
    and composable pipelines.
    """

    @abc.abstractmethod
    def handle(self, input: IN, **kwargs: Any) -> AsyncIterator[OUT]:
        """Process input and yield outputs.

        Args:
            input: The input to process.
            **kwargs: Additional context passed through the pipeline.

        Yields:
            Processed output items.
        """
        ...


class AgentHandler(Handler[str, RunResult]):
    """Routes execution between agents in a swarm with topology-aware stops.

    Manages agent dispatch, handoff detection, and stop condition
    checks for workflow, handoff, and team modes.

    Args:
        agents: Dict mapping agent name to agent instance.
        mode: Swarm topology mode.
        flow_order: Ordered list of agent names for workflow mode.
        provider: LLM provider for agent execution.
        max_handoffs: Maximum handoff count before stopping (handoff mode).
    """

    def __init__(
        self,
        *,
        agents: dict[str, Any],
        mode: SwarmMode = SwarmMode.WORKFLOW,
        flow_order: list[str] | None = None,
        provider: Any = None,
        max_handoffs: int = 10,
    ) -> None:
        self.agents = agents
        self.mode = mode
        self.flow_order = flow_order or list(agents.keys())
        self.provider = provider
        self.max_handoffs = max_handoffs

    async def handle(self, input: str, **kwargs: Any) -> AsyncIterator[RunResult]:
        """Execute agents according to the swarm topology.

        For workflow mode, runs agents in flow_order sequentially.
        For handoff mode, starts with the first agent and follows
        handoff chains.  For team mode, runs the lead agent which
        can delegate to workers.

        Args:
            input: User query string.
            **kwargs: Additional context (messages, state, etc.).

        Yields:
            ``RunResult`` from each agent execution.
        """
        messages: list[Message] = list(kwargs.get("messages", []))
        state = RunState(agent_name=self.flow_order[0] if self.flow_order else "")

        if self.mode == SwarmMode.WORKFLOW:
            async for result in self._run_workflow(input, messages, state):
                yield result
        elif self.mode == SwarmMode.HANDOFF:
            async for result in self._run_handoff(input, messages, state):
                yield result
        elif self.mode == SwarmMode.TEAM:
            async for result in self._run_team(input, messages, state):
                yield result

    async def _run_workflow(
        self, input: str, messages: list[Message], state: RunState
    ) -> AsyncIterator[RunResult]:
        """Execute agents sequentially in flow order.

        Output of each agent becomes input for the next.
        """
        current_input = input
        for agent_name in self.flow_order:
            agent = self.agents.get(agent_name)
            if agent is None:
                raise HandlerError(f"Agent '{agent_name}' not found in swarm")
            result = await call_runner(
                agent, current_input, messages=messages, provider=self.provider
            )
            yield result
            current_input = result.output

    async def _run_handoff(
        self, input: str, messages: list[Message], state: RunState
    ) -> AsyncIterator[RunResult]:
        """Execute agents following handoff chains.

        Starts with the first agent; if the agent's output references
        a handoff target, control transfers to that agent.
        """
        current_agent_name = self.flow_order[0] if self.flow_order else ""
        current_input = input
        handoff_count = 0

        while current_agent_name:
            agent = self.agents.get(current_agent_name)
            if agent is None:
                raise HandlerError(f"Agent '{current_agent_name}' not found in swarm")

            result = await call_runner(
                agent, current_input, messages=messages, provider=self.provider
            )
            yield result

            # Check for handoff in the result
            next_agent = self._detect_handoff(agent, result)
            if next_agent is None:
                break

            handoff_count += 1
            if handoff_count >= self.max_handoffs:
                raise HandlerError(f"Max handoffs ({self.max_handoffs}) exceeded in swarm")

            current_agent_name = next_agent
            current_input = result.output

    async def _run_team(
        self, input: str, messages: list[Message], state: RunState
    ) -> AsyncIterator[RunResult]:
        """Execute team mode: lead agent delegates to workers.

        The first agent in flow_order is the lead.  Workers are
        available as the lead's handoff targets.
        """
        if not self.flow_order:
            raise HandlerError("Team mode requires at least one agent")

        lead_name = self.flow_order[0]
        lead = self.agents.get(lead_name)
        if lead is None:
            raise HandlerError(f"Lead agent '{lead_name}' not found in swarm")

        # Run the lead agent — it can delegate to workers via handoffs
        result = await call_runner(lead, input, messages=messages, provider=self.provider)
        yield result

    def _detect_handoff(self, agent: Any, result: RunResult) -> str | None:
        """Check if the agent's result indicates a handoff.

        Looks for a handoff target name in the result output that
        matches one of the agent's declared handoff targets.

        Args:
            agent: The agent that produced the result.
            result: The run result to check.

        Returns:
            The target agent name, or None if no handoff detected.
        """
        handoffs: dict[str, Any] = getattr(agent, "handoffs", {})
        if not handoffs:
            return None

        output = result.output.strip()
        # Check if the output matches a handoff target name
        for target_name in handoffs:
            if target_name in self.agents and output == target_name:
                return target_name

        return None

    def _check_workflow_stop(self, agent_name: str) -> bool:
        """Check if the workflow should stop after this agent.

        Returns True if agent_name is the last in flow_order.
        """
        if not self.flow_order:
            return True
        return agent_name == self.flow_order[-1]

    def _check_handoff_stop(self, result: RunResult, agent: Any) -> bool:
        """Check if the handoff chain should stop.

        Returns True if no handoff target is detected.
        """
        return self._detect_handoff(agent, result) is None

    def _check_team_stop(self, agent_name: str) -> bool:
        """Check if team execution should stop.

        In team mode, execution stops after the lead agent completes.
        """
        if not self.flow_order:
            return True
        return agent_name == self.flow_order[0]
