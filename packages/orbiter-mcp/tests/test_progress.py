"""Tests for MCP progress notification capture (US-028)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, TextContent  # pyright: ignore[reportMissingImports]
from mcp.types import Tool as MCPTool  # pyright: ignore[reportMissingImports]

from orbiter.mcp.client import (  # pyright: ignore[reportMissingImports]
    MCPClientError,
    MCPServerConfig,
    MCPServerConnection,
    MCPTransport,
)
from orbiter.mcp.tools import (  # pyright: ignore[reportMissingImports]
    MCPToolWrapper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool(
    name: str = "search",
    description: str = "Search tool",
) -> MCPTool:
    return MCPTool(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
    )


def _make_call_result(text: str = "result") -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


# ---------------------------------------------------------------------------
# MCPProgressEvent type tests
# ---------------------------------------------------------------------------


class TestMCPProgressEvent:
    """MCPProgressEvent is a proper Pydantic model with the right fields."""

    def test_basic_construction(self) -> None:
        from orbiter.types import MCPProgressEvent

        evt = MCPProgressEvent(tool_name="my_tool", progress=5, total=10, message="halfway")
        assert evt.tool_name == "my_tool"
        assert evt.progress == 5
        assert evt.total == 10
        assert evt.message == "halfway"
        assert evt.type == "mcp_progress"

    def test_default_fields(self) -> None:
        from orbiter.types import MCPProgressEvent

        evt = MCPProgressEvent(tool_name="t", progress=3)
        assert evt.total is None
        assert evt.message == ""
        assert evt.agent_name == ""

    def test_is_in_stream_event_union(self) -> None:
        """MCPProgressEvent must be part of the StreamEvent type union."""
        from orbiter.types import MCPProgressEvent, StreamEvent

        evt = MCPProgressEvent(tool_name="t", progress=1)
        # StreamEvent is a Union type alias; verify isinstance works with each member
        # The simplest check: the annotation includes MCPProgressEvent
        import typing

        args = typing.get_args(StreamEvent)
        assert MCPProgressEvent in args

    def test_frozen_model(self) -> None:
        from orbiter.types import MCPProgressEvent

        evt = MCPProgressEvent(tool_name="t", progress=1)
        with pytest.raises(Exception):
            evt.progress = 99  # type: ignore[misc]

    def test_progress_is_int(self) -> None:
        from orbiter.types import MCPProgressEvent

        evt = MCPProgressEvent(tool_name="t", progress=7, total=20)
        assert isinstance(evt.progress, int)
        assert isinstance(evt.total, int)


# ---------------------------------------------------------------------------
# MCPServerConnection.call_tool progress_callback tests
# ---------------------------------------------------------------------------


class TestMCPServerConnectionProgressCallback:
    """MCPServerConnection.call_tool() passes progress_callback to the session."""

    @pytest.mark.asyncio
    async def test_progress_callback_passed_to_session(self) -> None:
        config = MCPServerConfig(
            name="test-server",
            transport=MCPTransport.STDIO,
            command="echo",
        )
        conn = MCPServerConnection(config)

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=_make_call_result("ok"))
        conn._session = mock_session

        received: list[Any] = []

        async def my_callback(progress: float, total: float | None, message: str | None) -> None:
            received.append((progress, total, message))

        await conn.call_tool("search", {"q": "test"}, progress_callback=my_callback)

        mock_session.call_tool.assert_awaited_once()
        call_kwargs = mock_session.call_tool.call_args
        assert call_kwargs.kwargs.get("progress_callback") is my_callback

    @pytest.mark.asyncio
    async def test_no_progress_callback_default_none(self) -> None:
        config = MCPServerConfig(name="s", transport=MCPTransport.STDIO, command="echo")
        conn = MCPServerConnection(config)

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=_make_call_result("ok"))
        conn._session = mock_session

        await conn.call_tool("t", {"a": 1})

        call_kwargs = mock_session.call_tool.call_args
        # progress_callback should be None (default)
        assert call_kwargs.kwargs.get("progress_callback") is None

    @pytest.mark.asyncio
    async def test_not_connected_raises(self) -> None:
        config = MCPServerConfig(name="s", transport=MCPTransport.STDIO, command="echo")
        conn = MCPServerConnection(config)
        with pytest.raises(MCPClientError, match="not connected"):
            await conn.call_tool("t", None, progress_callback=None)


# ---------------------------------------------------------------------------
# MCPToolWrapper.progress_queue tests
# ---------------------------------------------------------------------------


class TestMCPToolWrapperProgressQueue:
    """MCPToolWrapper has a progress_queue and populates it during execute()."""

    def test_progress_queue_created_in_init(self) -> None:
        mcp_tool = _make_mcp_tool()
        call_fn = AsyncMock(return_value=_make_call_result("ok"))
        wrapper = MCPToolWrapper(mcp_tool, "srv", call_fn)
        assert hasattr(wrapper, "progress_queue")
        assert isinstance(wrapper.progress_queue, asyncio.Queue)

    def test_progress_queue_created_in_from_dict(self) -> None:
        mcp_tool = _make_mcp_tool()
        call_fn = AsyncMock(return_value=_make_call_result("ok"))
        wrapper = MCPToolWrapper(mcp_tool, "srv", call_fn)
        data = wrapper.to_dict()
        # Manually add server_config so from_dict can reconstruct
        reconstructed = MCPToolWrapper.from_dict(
            {
                "__mcp_tool__": True,
                "name": "mcp__srv__search",
                "description": "Search tool",
                "parameters": {"type": "object", "properties": {}},
                "original_name": "search",
                "server_name": "srv",
                "large_output": False,
            }
        )
        assert hasattr(reconstructed, "progress_queue")
        assert isinstance(reconstructed.progress_queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_execute_populates_progress_queue(self) -> None:
        """When call_fn triggers the progress callback, MCPProgressEvent items go in the queue."""
        from orbiter.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool()
        captured_callbacks: list[Any] = []

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            captured_callbacks.append(progress_callback)
            # Simulate the server firing two progress notifications
            if progress_callback is not None:
                await progress_callback(1.0, 3.0, "step 1")
                await progress_callback(2.0, 3.0, "step 2")
            return _make_call_result("final result")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        result = await wrapper.execute(q="hello")

        # Final result is returned, NOT the progress events
        assert result == "final result"

        # Progress queue contains the two events
        assert wrapper.progress_queue.qsize() == 2

        evt1: MCPProgressEvent = wrapper.progress_queue.get_nowait()
        assert evt1.progress == 1
        assert evt1.total == 3
        assert evt1.message == "step 1"

        evt2: MCPProgressEvent = wrapper.progress_queue.get_nowait()
        assert evt2.progress == 2
        assert evt2.total == 3
        assert evt2.message == "step 2"

    @pytest.mark.asyncio
    async def test_execute_no_progress_queue_empty(self) -> None:
        """When no progress notifications fire, progress_queue stays empty."""
        mcp_tool = _make_mcp_tool()
        call_fn = AsyncMock(return_value=_make_call_result("ok"))
        wrapper = MCPToolWrapper(mcp_tool, "srv", call_fn)
        result = await wrapper.execute(q="hi")
        assert result == "ok"
        assert wrapper.progress_queue.empty()

    @pytest.mark.asyncio
    async def test_execute_result_not_in_progress_queue(self) -> None:
        """The tool result string is never placed in the progress_queue."""
        from orbiter.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool()

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(50.0, 100.0, "halfway")
            return _make_call_result("big result")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        result = await wrapper.execute(q="x")

        assert result == "big result"
        assert wrapper.progress_queue.qsize() == 1
        # The queued item is MCPProgressEvent, not a string
        item = wrapper.progress_queue.get_nowait()
        assert isinstance(item, MCPProgressEvent)
        assert item.message == "halfway"

    @pytest.mark.asyncio
    async def test_progress_event_total_none_when_unknown(self) -> None:
        """When MCP total is None, MCPProgressEvent.total is also None."""
        from orbiter.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool()

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(5.0, None, "processing...")
            return _make_call_result("done")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        await wrapper.execute()

        evt: MCPProgressEvent = wrapper.progress_queue.get_nowait()
        assert evt.progress == 5
        assert evt.total is None
        assert evt.message == "processing..."

    @pytest.mark.asyncio
    async def test_progress_event_message_none_becomes_empty_string(self) -> None:
        """When MCP message is None, MCPProgressEvent.message is ''."""
        from orbiter.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool()

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(1.0, 5.0, None)
            return _make_call_result("done")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        await wrapper.execute()

        evt: MCPProgressEvent = wrapper.progress_queue.get_nowait()
        assert evt.message == ""

    @pytest.mark.asyncio
    async def test_progress_queue_accumulates_across_calls(self) -> None:
        """Progress queue accumulates items; runner must drain between calls."""
        from orbiter.types import MCPProgressEvent

        mcp_tool = _make_mcp_tool()

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            if progress_callback is not None:
                await progress_callback(1.0, 2.0, "first")
            return _make_call_result("done")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)
        await wrapper.execute()
        await wrapper.execute()

        # Both calls contribute to the same queue
        assert wrapper.progress_queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_progress_callback_not_passed_to_session_when_import_fails(self) -> None:
        """When orbiter-core MCPProgressEvent is not importable, no callback is passed."""
        mcp_tool = _make_mcp_tool()
        captured: list[Any] = []

        async def mock_call_fn(
            name: str, args: Any, *, progress_callback: Any = None
        ) -> CallToolResult:
            captured.append(progress_callback)
            return _make_call_result("ok")

        wrapper = MCPToolWrapper(mcp_tool, "srv", mock_call_fn)

        import sys
        orig = sys.modules.get("orbiter.types")
        sys.modules["orbiter.types"] = None  # type: ignore[assignment]
        try:
            result = await wrapper.execute(q="test")
        finally:
            if orig is None:
                del sys.modules["orbiter.types"]
            else:
                sys.modules["orbiter.types"] = orig

        assert result == "ok"
        # Callback should be None when import fails
        assert captured[0] is None
