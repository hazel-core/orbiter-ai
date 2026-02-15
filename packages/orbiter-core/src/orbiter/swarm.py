"""Swarm: multi-agent orchestration with flow DSL.

A ``Swarm`` groups multiple agents and defines their execution
topology using a simple DSL (``"a >> b >> c"``).  Currently
supports ``mode='workflow'`` which runs agents sequentially,
passing each agent's output as the next agent's input.

Usage::

    swarm = Swarm(
        agents=[agent_a, agent_b, agent_c],
        flow="a >> b >> c",
    )
    result = await run(swarm, "Hello!")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from orbiter._internal.call_runner import call_runner
from orbiter._internal.graph import GraphError, parse_flow_dsl, topological_sort
from orbiter.types import Message, OrbiterError, RunResult


class SwarmError(OrbiterError):
    """Raised for swarm-level errors (invalid flow, missing agents, etc.)."""


class Swarm:
    """Multi-agent orchestration container.

    Groups agents and defines their execution topology via a flow DSL.
    In workflow mode, agents run sequentially with output→input chaining.

    Args:
        agents: List of ``Agent`` instances to include in the swarm.
        flow: Flow DSL string defining execution order
            (e.g., ``"a >> b >> c"``).  If not provided, agents
            run in the order they are given.
        mode: Execution mode.  Currently only ``"workflow"`` is
            supported.
    """

    def __init__(
        self,
        *,
        agents: list[Any],
        flow: str | None = None,
        mode: str = "workflow",
    ) -> None:
        if not agents:
            raise SwarmError("Swarm requires at least one agent")

        self.mode = mode

        # Index agents by name for O(1) lookup
        self.agents: dict[str, Any] = {}
        for agent in agents:
            name = agent.name
            if name in self.agents:
                raise SwarmError(f"Duplicate agent name '{name}' in swarm")
            self.agents[name] = agent

        # Resolve execution order from flow DSL or agent list order
        if flow is not None:
            try:
                graph = parse_flow_dsl(flow)
            except GraphError as exc:
                raise SwarmError(f"Invalid flow DSL: {exc}") from exc

            # Validate all flow nodes are known agents
            for node_name in graph.nodes:
                if node_name not in self.agents:
                    raise SwarmError(f"Flow references unknown agent '{node_name}'")

            try:
                self.flow_order = topological_sort(graph)
            except GraphError as exc:
                raise SwarmError(f"Cycle in flow DSL: {exc}") from exc
        else:
            # Default: run in the order agents were provided
            self.flow_order = [a.name for a in agents]

        self.flow = flow

        # Set name from the first agent for compatibility with runner
        self.name = f"swarm({self.flow_order[0]}...)"

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute the swarm according to its mode.

        In workflow mode, agents execute in topological order.
        Each agent's output becomes the next agent's input.

        Args:
            input: User query string.
            messages: Prior conversation history.
            provider: LLM provider for all agents.
            max_retries: Retry attempts for transient errors.

        Returns:
            ``RunResult`` from the final agent in the chain.

        Raises:
            SwarmError: If mode is unsupported or an agent fails.
        """
        if self.mode == "workflow":
            return await self._run_workflow(
                input, messages=messages, provider=provider, max_retries=max_retries
            )
        raise SwarmError(f"Unsupported swarm mode: {self.mode!r}")

    async def _run_workflow(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute agents sequentially, chaining output→input.

        Returns the ``RunResult`` from the last agent in the flow.
        """
        current_input = input
        last_result: RunResult | None = None

        for agent_name in self.flow_order:
            agent = self.agents[agent_name]
            last_result = await call_runner(
                agent,
                current_input,
                messages=messages,
                provider=provider,
                max_retries=max_retries,
            )
            current_input = last_result.output

        assert last_result is not None  # guaranteed since agents is non-empty
        return last_result

    def describe(self) -> dict[str, Any]:
        """Return a summary of the swarm's configuration.

        Returns:
            Dict with mode, flow order, and agent descriptions.
        """
        return {
            "mode": self.mode,
            "flow": self.flow,
            "flow_order": self.flow_order,
            "agents": {name: agent.describe() for name, agent in self.agents.items()},
        }

    def __repr__(self) -> str:
        return f"Swarm(mode={self.mode!r}, agents={list(self.agents.keys())}, flow={self.flow!r})"
