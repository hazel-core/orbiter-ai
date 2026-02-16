"""Swarm: multi-agent orchestration with flow DSL.

A ``Swarm`` groups multiple agents and defines their execution
topology using a simple DSL (``"a >> b >> c"``).  Supports
``mode='workflow'`` (sequential pipeline), ``mode='handoff'``
(agent-driven delegation), and ``mode='team'`` (lead-worker
delegation).

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
from orbiter.tool import Tool
from orbiter.types import Message, OrbiterError, RunResult


class SwarmError(OrbiterError):
    """Raised for swarm-level errors (invalid flow, missing agents, etc.)."""


class Swarm:
    """Multi-agent orchestration container.

    Groups agents and defines their execution topology via a flow DSL.
    In workflow mode, agents run sequentially with output→input chaining.
    In handoff mode, agents delegate dynamically via handoff targets.
    In team mode, the first agent is the lead and others are workers;
    the lead can delegate to workers via auto-generated tools.

    Args:
        agents: List of ``Agent`` instances to include in the swarm.
        flow: Flow DSL string defining execution order
            (e.g., ``"a >> b >> c"``).  If not provided, agents
            run in the order they are given.
        mode: Execution mode — ``"workflow"``, ``"handoff"``, or ``"team"``.
        max_handoffs: Maximum number of handoff transitions before
            raising an error (handoff mode only).
    """

    def __init__(
        self,
        *,
        agents: list[Any],
        flow: str | None = None,
        mode: str = "workflow",
        max_handoffs: int = 10,
    ) -> None:
        if not agents:
            raise SwarmError("Swarm requires at least one agent")

        self.mode = mode
        self.max_handoffs = max_handoffs

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
        In handoff mode, the first agent runs and can hand off
        to other agents dynamically.

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
        if self.mode == "handoff":
            return await self._run_handoff(
                input, messages=messages, provider=provider, max_retries=max_retries
            )
        if self.mode == "team":
            return await self._run_team(
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

        Supports both regular agents and group nodes (``ParallelGroup``,
        ``SerialGroup``).  Groups have an ``is_group`` attribute and
        their own ``run()`` method.

        Returns the ``RunResult`` from the last agent in the flow.
        """
        current_input = input
        last_result: RunResult | None = None

        for agent_name in self.flow_order:
            agent = self.agents[agent_name]

            if getattr(agent, "is_group", False) or getattr(agent, "is_swarm", False):
                last_result = await agent.run(
                    current_input,
                    messages=messages,
                    provider=provider,
                    max_retries=max_retries,
                )
            else:
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

    async def _run_handoff(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute agents following handoff chains.

        Starts with the first agent in flow_order.  If an agent's
        output matches a handoff target name (declared on the agent),
        control transfers to that target with the full conversation
        history.  Stops when an agent produces output that is not a
        handoff target, or when ``max_handoffs`` is exceeded.

        Returns the ``RunResult`` from the last agent that ran.
        """
        current_agent_name = self.flow_order[0]
        current_input = input
        all_messages: list[Message] = list(messages) if messages else []
        handoff_count = 0

        while True:
            agent = self.agents[current_agent_name]
            result = await call_runner(
                agent,
                current_input,
                messages=all_messages,
                provider=provider,
                max_retries=max_retries,
            )

            # Accumulate conversation history from this agent's run
            all_messages = list(result.messages)

            # Check for handoff
            next_agent = self._detect_handoff(agent, result)
            if next_agent is None:
                return result

            handoff_count += 1
            if handoff_count > self.max_handoffs:
                raise SwarmError(f"Max handoffs ({self.max_handoffs}) exceeded in swarm")

            current_agent_name = next_agent
            current_input = result.output

    def _detect_handoff(self, agent: Any, result: RunResult) -> str | None:
        """Check if an agent's result indicates a handoff.

        Matches the agent's output (stripped) against its declared
        handoff target names.  The target must also exist in the
        swarm's agents dict.

        Returns:
            Target agent name, or ``None`` if no handoff detected.
        """
        handoffs: dict[str, Any] = getattr(agent, "handoffs", {})
        if not handoffs:
            return None

        output = result.output.strip()
        for target_name in handoffs:
            if target_name in self.agents and output == target_name:
                return target_name

        return None

    async def _run_team(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute team mode: lead delegates to workers via tools.

        The first agent in flow_order is the lead.  Other agents are
        workers.  The lead receives auto-generated ``delegate_to_{name}``
        tools that invoke worker agents.  When the lead calls a delegate
        tool, the worker runs and its output is returned as the tool
        result.  The lead then synthesizes the final output.

        Returns the ``RunResult`` from the lead agent.
        """
        if len(self.agents) < 2:
            raise SwarmError("Team mode requires at least two agents (lead + workers)")

        lead_name = self.flow_order[0]
        lead = self.agents[lead_name]
        worker_names = [n for n in self.flow_order if n != lead_name]

        # Create delegate tools for each worker and add to lead
        delegate_tools: list[Tool] = []
        for worker_name in worker_names:
            worker = self.agents[worker_name]
            dtool = _DelegateTool(
                worker=worker,
                provider=provider,
                max_retries=max_retries,
            )
            delegate_tools.append(dtool)

        # Temporarily add delegate tools to the lead agent
        original_tools = dict(lead.tools)
        for dtool in delegate_tools:
            lead.tools[dtool.name] = dtool

        try:
            result = await call_runner(
                lead,
                input,
                messages=messages,
                provider=provider,
                max_retries=max_retries,
            )
        finally:
            # Restore original tools
            lead.tools = original_tools

        return result

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


class _DelegateTool(Tool):
    """Auto-generated tool that delegates work to a worker agent.

    When the lead agent calls this tool, the worker agent runs with
    the provided task description and its output is returned as the
    tool result.
    """

    def __init__(
        self,
        *,
        worker: Any,
        provider: Any = None,
        max_retries: int = 3,
    ) -> None:
        worker_name: str = worker.name
        self.name = f"delegate_to_{worker_name}"
        self.description = f"Delegate a task to the '{worker_name}' worker agent."
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": f"The task description to send to '{worker_name}'.",
                },
            },
            "required": ["task"],
        }
        self._worker = worker
        self._provider = provider
        self._max_retries = max_retries

    async def execute(self, **kwargs: Any) -> str:
        """Run the worker agent with the given task.

        Args:
            **kwargs: Must include ``task`` (str).

        Returns:
            The worker agent's output text.
        """
        task: str = kwargs.get("task", "")
        result = await call_runner(
            self._worker,
            task,
            provider=self._provider,
            max_retries=self._max_retries,
        )
        return result.output
