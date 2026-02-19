"""Automatically persist LLM responses and tool results to a memory store.

Registers POST_LLM_CALL and POST_TOOL_CALL hooks on an agent so that
conversation turns are saved without manual intervention.  The caller
saves a ``HumanMemory`` before calling ``run()`` / ``run.stream()``.
"""

from __future__ import annotations

from typing import Any

from orbiter.hooks import HookPoint  # pyright: ignore[reportMissingImports]
from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
    AIMemory,
    MemoryMetadata,
    MemoryStore,
    ToolMemory,
)


class MemoryPersistence:
    """Hook-based auto-persistence for conversation memory.

    Attaches ``POST_LLM_CALL`` and ``POST_TOOL_CALL`` hooks to an agent
    that automatically save ``AIMemory`` and ``ToolMemory`` items to the
    provided store.

    Args:
        store: A :class:`MemoryStore` implementation to persist items to.
        metadata: Optional scoping metadata applied to every saved item.
    """

    def __init__(
        self,
        store: MemoryStore,
        metadata: MemoryMetadata | None = None,
    ) -> None:
        self.store = store
        self.metadata = metadata or MemoryMetadata()

    async def _save_llm_response(self, *, agent: Any, response: Any, **_: Any) -> None:
        """POST_LLM_CALL hook — save an AIMemory item."""
        tool_calls = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in getattr(response, "tool_calls", []) or []
        ]
        await self.store.add(
            AIMemory(
                content=getattr(response, "content", "") or "",
                metadata=self.metadata,
                tool_calls=tool_calls,
            )
        )

    async def _save_tool_result(self, *, agent: Any, tool_name: str, result: Any, **_: Any) -> None:
        """POST_TOOL_CALL hook — save a ToolMemory item."""
        await self.store.add(
            ToolMemory(
                content=result.content or result.error or "",
                metadata=self.metadata,
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                is_error=result.error is not None,
            )
        )

    def attach(self, agent: Any) -> None:
        """Register persistence hooks on the given agent."""
        agent.hook_manager.add(HookPoint.POST_LLM_CALL, self._save_llm_response)
        agent.hook_manager.add(HookPoint.POST_TOOL_CALL, self._save_tool_result)

    def detach(self, agent: Any) -> None:
        """Remove persistence hooks from the given agent."""
        agent.hook_manager.remove(HookPoint.POST_LLM_CALL, self._save_llm_response)
        agent.hook_manager.remove(HookPoint.POST_TOOL_CALL, self._save_tool_result)
