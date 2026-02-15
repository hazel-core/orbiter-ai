"""Core execution loop for running an agent with state tracking.

The call runner orchestrates the LLM→tool→LLM cycle, integrating
with ``RunState`` for message tracking, node lifecycle, and usage
accumulation.  It also detects endless loops where the agent
repeatedly produces the same tool calls without making progress.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from orbiter._internal.message_builder import build_messages
from orbiter._internal.state import RunState
from orbiter.types import (
    AgentOutput,
    Message,
    OrbiterError,
    RunResult,
)


class CallRunnerError(OrbiterError):
    """Raised for call runner errors (loop detection, state errors)."""


async def call_runner(
    agent: Any,
    input: str,
    *,
    state: RunState | None = None,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult:
    """Run an agent's full LLM-tool loop with state tracking.

    Orchestrates message building, LLM calls, tool execution, and
    result aggregation via ``Agent.run()``.  Wraps each step in a
    ``RunNode`` for lifecycle tracking and detects endless loops
    where the same tool calls repeat without progress.

    Args:
        agent: An ``Agent`` instance with a ``run()`` method.
        input: User query string.
        state: Optional pre-existing ``RunState``.  A new one is
            created if not provided.
        messages: Prior conversation history.
        provider: LLM provider with an ``async complete()`` method.
        max_retries: Retry attempts passed to ``Agent.run()``.
        loop_threshold: Number of consecutive identical tool-call
            patterns before raising ``CallRunnerError``.

    Returns:
        ``RunResult`` with the final output, full message history,
        aggregated usage, and step count.

    Raises:
        CallRunnerError: On endless loop detection or agent failure.
    """
    if state is None:
        state = RunState(agent_name=agent.name)

    state.start()
    node = state.new_node(agent_name=agent.name)
    node.start()

    try:
        output = await agent.run(
            input,
            messages=messages,
            provider=provider,
            max_retries=max_retries,
        )

        # Record usage and mark success
        if output.usage:
            state.record_usage(output.usage)
            node.succeed(usage=output.usage)
        else:
            node.succeed()

        # Detect endless loops by checking repeated tool-call patterns
        _check_loop(state, output, loop_threshold)

        # Build final message list for the result
        instr: str = ""
        raw_instr = agent.instructions
        if callable(raw_instr):
            instr = str(raw_instr(agent.name))
        elif raw_instr:
            instr = str(raw_instr)
        final_messages = build_messages(
            instr,
            list(messages) if messages else [],
        )
        state.add_messages(final_messages)

        state.succeed()

        return RunResult(
            output=output.text,
            messages=state.messages,
            usage=state.total_usage,
            steps=state.iterations,
        )

    except CallRunnerError:
        node.fail("Endless loop detected")
        state.fail("Endless loop detected")
        raise

    except Exception as exc:
        error_msg = str(exc)
        node.fail(error_msg)
        state.fail(error_msg)
        raise CallRunnerError(f"Call runner failed for agent '{agent.name}': {error_msg}") from exc


def _check_loop(
    state: RunState,
    output: AgentOutput,
    threshold: int,
) -> None:
    """Detect endless loops by checking for repeated tool-call patterns.

    If the agent produces the same set of tool calls ``threshold``
    times consecutively (tracked via node metadata), a
    ``CallRunnerError`` is raised.

    Args:
        state: Current run state with node history.
        output: The latest agent output.
        threshold: How many consecutive repeats trigger the error.
    """
    if not output.tool_calls:
        return

    # Build a signature from tool names + arguments
    signature = _tool_call_signature(output)

    # Check how many recent consecutive nodes share this signature
    consecutive = 0
    for node in reversed(state.nodes):
        if node.metadata.get("tool_signature") == signature:
            consecutive += 1
        else:
            break

    # Store the signature on the current node for future checks
    current = state.current_node
    if current is not None:
        current.metadata["tool_signature"] = signature

    if consecutive >= threshold:
        raise CallRunnerError(
            f"Endless loop detected: same tool calls repeated "
            f"{consecutive} times (threshold={threshold})"
        )


def _tool_call_signature(output: AgentOutput) -> str:
    """Create a deterministic string signature for tool calls.

    Sorts by tool name to be order-independent, then combines
    names and argument strings.

    Args:
        output: Agent output containing tool calls.

    Returns:
        A string signature for comparison.
    """
    parts = sorted(f"{tc.name}:{tc.arguments}" for tc in output.tool_calls)
    return "|".join(parts)
