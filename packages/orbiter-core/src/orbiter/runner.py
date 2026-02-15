"""Public entry point for running agents.

Provides ``run()`` (async) and ``run.sync()`` (blocking) as the
primary API for executing an ``Agent``.  Internally delegates to
:func:`orbiter._internal.call_runner.call_runner` for state tracking
and loop detection.

Usage::

    result = await run(agent, "Hello!")
    result = run.sync(agent, "Hello!")
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from orbiter._internal.call_runner import call_runner
from orbiter.types import Message, RunResult


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


# Attach sync as an attribute of the run function
run.sync = _sync  # type: ignore[attr-defined]
