"""Public entry point for running agents.

Provides ``run()`` (async), ``run.sync()`` (blocking), and
``run.stream()`` (async generator) as the primary API for executing
an ``Agent``.  Internally delegates to
:func:`orbiter._internal.call_runner.call_runner` for state tracking
and loop detection.

Usage::

    result = await run(agent, "Hello!")
    result = run.sync(agent, "Hello!")
    async for event in run.stream(agent, "Hello!"):
        print(event)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from orbiter._internal.call_runner import call_runner
from orbiter._internal.message_builder import build_messages
from orbiter._internal.output_parser import parse_tool_arguments
from orbiter.types import (
    AssistantMessage,
    Message,
    RunResult,
    StreamEvent,
    TextEvent,
    ToolCall,
    ToolCallEvent,
    UserMessage,
)


async def run(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult:
    """Execute an agent (or swarm) and return the result.

    This is the primary async API for running agents.  For a blocking
    variant, use ``run.sync()``.

    If *provider* is ``None``, a default provider is resolved from the
    agent's ``provider_name`` using the model registry (if available).

    Args:
        agent: An ``Agent`` (or ``Swarm``) instance.
        input: User query string.
        messages: Prior conversation history to continue from.
        provider: LLM provider with ``async complete()`` method.
            When ``None``, auto-resolved from the agent's model string.
        max_retries: Retry attempts for transient LLM errors.
        loop_threshold: Consecutive identical tool-call patterns
            before raising a loop error.

    Returns:
        ``RunResult`` with the agent's output, message history,
        usage stats, and step count.
    """
    resolved_provider = provider or _resolve_provider(agent)
    return await call_runner(
        agent,
        input,
        messages=messages,
        provider=resolved_provider,
        max_retries=max_retries,
        loop_threshold=loop_threshold,
    )


def _sync(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult:
    """Execute an agent synchronously (blocking wrapper).

    Calls ``run()`` via ``asyncio.run()``.  This is a convenience for
    scripts and notebooks where an event loop is not already running.

    Args:
        agent: An ``Agent`` (or ``Swarm``) instance.
        input: User query string.
        messages: Prior conversation history to continue from.
        provider: LLM provider with ``async complete()`` method.
        max_retries: Retry attempts for transient LLM errors.
        loop_threshold: Consecutive identical tool-call patterns
            before raising a loop error.

    Returns:
        ``RunResult`` with the agent's output, message history,
        usage stats, and step count.
    """
    return asyncio.run(
        run(
            agent,
            input,
            messages=messages,
            provider=provider,
            max_retries=max_retries,
            loop_threshold=loop_threshold,
        )
    )


async def _stream(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_steps: int | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream agent execution, yielding events in real-time.

    Uses the provider's ``stream()`` method to deliver text deltas
    as ``TextEvent`` objects and emit ``ToolCallEvent`` for each tool
    invocation. When tool calls are detected, tools are executed and
    the LLM is re-streamed with the results — looping until a
    text-only response or *max_steps* is reached.

    Args:
        agent: An ``Agent`` instance.
        input: User query string.
        messages: Prior conversation history.
        provider: LLM provider with an ``async stream()`` method.
            When ``None``, auto-resolved from the agent's model string.
        max_steps: Maximum LLM-tool round-trips. Defaults to
            ``agent.max_steps``.

    Yields:
        ``TextEvent`` for text chunks and ``ToolCallEvent`` for tool
        invocations.
    """
    resolved = provider or _resolve_provider(agent)
    if resolved is None:
        from orbiter.agent import AgentError

        raise AgentError(f"Agent '{agent.name}' requires a provider for stream()")

    steps = max_steps if max_steps is not None else agent.max_steps

    # Resolve instructions
    instr: str = ""
    raw_instr = agent.instructions
    if callable(raw_instr):
        instr = str(raw_instr(agent.name))
    elif raw_instr:
        instr = str(raw_instr)

    # Build initial message list
    history: list[Message] = list(messages) if messages else []
    history.append(UserMessage(content=input))
    msg_list = build_messages(instr, history)

    tool_schemas = agent.get_tool_schemas() or None

    for _step in range(steps):
        # Accumulate text and tool call deltas from the stream
        text_parts: list[str] = []
        # dict of index -> accumulated tool call data
        tc_acc: dict[int, dict[str, str]] = {}

        async for chunk in resolved.stream(
            msg_list,
            tools=tool_schemas,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        ):
            # Yield text deltas
            if chunk.delta:
                text_parts.append(chunk.delta)
                yield TextEvent(text=chunk.delta, agent_name=agent.name)

            # Accumulate tool call deltas
            for tcd in chunk.tool_call_deltas:
                idx = tcd.index
                if idx not in tc_acc:
                    tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                if tcd.id is not None:
                    tc_acc[idx]["id"] = tcd.id
                if tcd.name is not None:
                    tc_acc[idx]["name"] = tcd.name
                tc_acc[idx]["arguments"] += tcd.arguments

        # Build completed tool calls
        tool_calls = [
            ToolCall(
                id=data["id"],
                name=data["name"],
                arguments=data["arguments"],
            )
            for data in tc_acc.values()
            if data["id"]
        ]

        # No tool calls — done streaming
        if not tool_calls:
            return

        # Yield ToolCallEvent for each tool call
        for tc in tool_calls:
            yield ToolCallEvent(
                tool_name=tc.name,
                tool_call_id=tc.id,
                agent_name=agent.name,
            )

        # Execute tools and feed results back
        full_text = "".join(text_parts)
        actions = parse_tool_arguments(tool_calls)
        tool_results = await agent._execute_tools(actions)

        # Append assistant message + tool results to conversation
        msg_list.append(AssistantMessage(content=full_text, tool_calls=tool_calls))
        msg_list.extend(tool_results)


def _resolve_provider(agent: Any) -> Any:
    """Attempt to auto-resolve a provider from the agent's model config.

    Tries the model registry from ``orbiter.models`` if available.
    Returns ``None`` if auto-resolution fails (call_runner will then
    let Agent.run() raise its own error for missing provider).
    """
    try:
        from orbiter.models.provider import get_provider  # pyright: ignore[reportMissingImports]

        return get_provider(agent.provider_name)
    except Exception:
        return None


# Attach sync and stream as attributes of the run function
run.sync = _sync  # type: ignore[attr-defined]
run.stream = _stream  # type: ignore[attr-defined]
