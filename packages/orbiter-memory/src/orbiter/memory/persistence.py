"""Automatically persist LLM responses and tool results to a memory store.

Registers POST_LLM_CALL and POST_TOOL_CALL hooks on an agent so that
conversation turns are saved without manual intervention.  The caller
saves a ``HumanMemory`` before calling ``run()`` / ``run.stream()``.
"""

from __future__ import annotations

import logging
from typing import Any

from orbiter.hooks import HookPoint  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)
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
        self._attached_agent_ids: set[int] = set()

    async def _save_llm_response(self, *, agent: Any, response: Any, **_: Any) -> None:
        """POST_LLM_CALL hook — save an AIMemory item."""
        tool_calls = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in getattr(response, "tool_calls", []) or []
        ]
        content = getattr(response, "content", None)
        item = AIMemory(
            content=content if content is not None else "",
            metadata=self.metadata,
            tool_calls=tool_calls,
        )
        await self.store.add(item)
        logger.debug("persisted AIMemory id=%s tool_calls=%d", item.id, len(tool_calls))

    async def _save_tool_result(self, *, agent: Any, tool_name: str, result: Any, **_: Any) -> None:
        """POST_TOOL_CALL hook — save a ToolMemory item."""
        # Use explicit None checks so that empty-string content is not replaced
        # by the error field.
        content = result.content if result.content is not None else (result.error or "")
        item = ToolMemory(
            content=content,
            metadata=self.metadata,
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            is_error=result.error is not None,
        )
        await self.store.add(item)
        logger.debug("persisted ToolMemory id=%s tool=%s is_error=%s", item.id, tool_name, item.is_error)

    def attach(self, agent: Any) -> None:
        """Register persistence hooks on the given agent.

        Idempotent — a second call for the same agent object is a no-op.
        """
        agent_id = id(agent)
        if agent_id in self._attached_agent_ids:
            logger.debug("already attached to agent=%s, skipping", getattr(agent, "name", agent))
            return
        self._attached_agent_ids.add(agent_id)
        agent.hook_manager.add(HookPoint.POST_LLM_CALL, self._save_llm_response)
        agent.hook_manager.add(HookPoint.POST_TOOL_CALL, self._save_tool_result)
        logger.debug("attached memory persistence hooks to agent=%s", getattr(agent, "name", agent))

    def detach(self, agent: Any) -> None:
        """Remove persistence hooks from the given agent."""
        self._attached_agent_ids.discard(id(agent))
        agent.hook_manager.remove(HookPoint.POST_LLM_CALL, self._save_llm_response)
        agent.hook_manager.remove(HookPoint.POST_TOOL_CALL, self._save_tool_result)
        logger.debug("detached memory persistence hooks from agent=%s", getattr(agent, "name", agent))
