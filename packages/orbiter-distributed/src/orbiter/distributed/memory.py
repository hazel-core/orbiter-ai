"""Memory helpers for distributed workers.

Provides store creation, teardown, and message conversion utilities.
All orbiter-memory imports are lazy (inside function bodies) to keep
the dependency optional.
"""

from __future__ import annotations

from typing import Any


async def create_memory_store(mem_config: dict[str, Any], task_id: str) -> tuple[Any, Any]:
    """Create a MemoryStore + MemoryMetadata from task metadata config.

    Config schema::

        {
            "backend": "short_term" | "sqlite" | "postgres",
            "dsn": "...",               # for sqlite/postgres
            "scope": {
                "user_id": "...",
                "session_id": "...",
                "agent_id": "...",
            },
        }

    Returns:
        Tuple of ``(store, metadata)``.
    """
    from orbiter.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]

    scope = mem_config.get("scope", {})
    metadata = MemoryMetadata(
        user_id=scope.get("user_id"),
        session_id=scope.get("session_id"),
        task_id=task_id,
        agent_id=scope.get("agent_id"),
    )

    backend = mem_config.get("backend", "short_term")

    if backend == "short_term":
        from orbiter.memory.short_term import (
            ShortTermMemory,  # pyright: ignore[reportMissingImports]
        )

        store = ShortTermMemory(scope="session")
        return store, metadata

    if backend == "sqlite":
        from orbiter.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
            SQLiteMemory,
        )

        dsn = mem_config.get("dsn", ":memory:")
        store = SQLiteMemory(dsn)
        return store, metadata

    if backend == "postgres":
        from orbiter.memory.backends.postgres import (  # pyright: ignore[reportMissingImports]
            PostgresMemory,
        )

        dsn = mem_config.get("dsn", "")
        store = PostgresMemory(dsn)
        return store, metadata

    # Fallback to short_term for unknown backends
    from orbiter.memory.short_term import ShortTermMemory  # pyright: ignore[reportMissingImports]

    store = ShortTermMemory(scope="session")
    return store, metadata


async def teardown_memory_store(store: Any) -> None:
    """Close store connections if a ``close()`` method exists."""
    close = getattr(store, "close", None)
    if close is not None and callable(close):
        result = close()
        # Support both sync and async close
        if hasattr(result, "__await__"):
            await result


def memory_items_to_messages(items: list[Any]) -> list[Any]:
    """Convert MemoryItems to typed Message objects.

    Mapping:
    - ``human`` → ``UserMessage``
    - ``ai`` → ``AssistantMessage``
    - ``tool`` → ``ToolResult``
    - ``system`` → ``SystemMessage``
    """
    from orbiter.types import (  # pyright: ignore[reportMissingImports]
        AssistantMessage,
        SystemMessage,
        ToolResult,
        UserMessage,
    )

    messages: list[Any] = []
    for item in items:
        mt = item.memory_type
        if mt == "human":
            messages.append(UserMessage(content=item.content))
        elif mt == "ai":
            tc = []
            if hasattr(item, "tool_calls") and item.tool_calls:
                from orbiter.types import ToolCall  # pyright: ignore[reportMissingImports]

                for call in item.tool_calls:
                    tc.append(
                        ToolCall(
                            id=call.get("id", ""),
                            name=call.get("name", ""),
                            arguments=call.get("arguments", ""),
                        )
                    )
            messages.append(AssistantMessage(content=item.content, tool_calls=tc))
        elif mt == "tool":
            messages.append(
                ToolResult(
                    tool_call_id=getattr(item, "tool_call_id", ""),
                    tool_name=getattr(item, "tool_name", ""),
                    content=item.content,
                    error=None if not getattr(item, "is_error", False) else item.content,
                )
            )
        elif mt == "system":
            messages.append(SystemMessage(content=item.content))
    return messages
