"""Tests for MCP tool schema extraction, conversion, and filtering."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, TextContent  # pyright: ignore[reportMissingImports]
from mcp.types import Tool as MCPTool  # pyright: ignore[reportMissingImports]

from orbiter.mcp.tools import (  # pyright: ignore[reportMissingImports]
    DEFAULT_NAMESPACE,
    MCPToolError,
    MCPToolFilter,
    MCPToolWrapper,
    convert_mcp_tools,
    extract_schema,
    load_tools_from_client,
    load_tools_from_connection,
    namespace_tool_name,
    parse_namespaced_name,
)
from orbiter.tool import Tool  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool(
    name: str = "test_tool",
    description: str = "A test tool",
    input_schema: dict[str, Any] | None = None,
) -> MCPTool:
    """Create a mock MCPTool."""
    schema = input_schema or {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }
    return MCPTool(name=name, description=description, inputSchema=schema)


def _make_call_result(text: str = "result", is_error: bool = False) -> CallToolResult:
    """Create a mock CallToolResult."""
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        isError=is_error,
    )


# ---------------------------------------------------------------------------
# MCPToolFilter tests
# ---------------------------------------------------------------------------


class TestMCPToolFilterInit:
    def test_defaults(self) -> None:
        f = MCPToolFilter()
        assert f.accepts("anything")

    def test_include_only(self) -> None:
        f = MCPToolFilter(include=["a", "b"])
        assert f.accepts("a")
        assert f.accepts("b")
        assert not f.accepts("c")

    def test_exclude_only(self) -> None:
        f = MCPToolFilter(exclude=["bad"])
        assert f.accepts("good")
        assert not f.accepts("bad")

    def test_exclude_takes_priority(self) -> None:
        f = MCPToolFilter(include=["a", "b"], exclude=["b"])
        assert f.accepts("a")
        assert not f.accepts("b")
        assert not f.accepts("c")

    def test_repr(self) -> None:
        f = MCPToolFilter(include=["b", "a"], exclude=["c"])
        r = repr(f)
        assert "MCPToolFilter" in r
        assert "['a', 'b']" in r
        assert "['c']" in r


class TestMCPToolFilterApply:
    def test_apply_empty_filter(self) -> None:
        tools = [_make_mcp_tool("a"), _make_mcp_tool("b")]
        result = MCPToolFilter().apply(tools)
        assert len(result) == 2

    def test_apply_include(self) -> None:
        tools = [_make_mcp_tool("a"), _make_mcp_tool("b"), _make_mcp_tool("c")]
        result = MCPToolFilter(include=["a", "c"]).apply(tools)
        assert [t.name for t in result] == ["a", "c"]

    def test_apply_exclude(self) -> None:
        tools = [_make_mcp_tool("a"), _make_mcp_tool("b"), _make_mcp_tool("c")]
        result = MCPToolFilter(exclude=["b"]).apply(tools)
        assert [t.name for t in result] == ["a", "c"]


# ---------------------------------------------------------------------------
# Namespace mapping tests
# ---------------------------------------------------------------------------


class TestNamespaceToolName:
    def test_basic(self) -> None:
        result = namespace_tool_name("search", "my_server")
        assert result == "mcp__my_server__search"

    def test_custom_namespace(self) -> None:
        result = namespace_tool_name("read", "fs", namespace="tool")
        assert result == "tool__fs__read"

    def test_sanitizes_special_chars(self) -> None:
        result = namespace_tool_name("file-read", "my-server.v2")
        assert result == "mcp__my_server_v2__file_read"

    def test_preserves_underscores(self) -> None:
        result = namespace_tool_name("get_data", "data_server")
        assert result == "mcp__data_server__get_data"


class TestParseNamespacedName:
    def test_basic(self) -> None:
        ns, server, tool = parse_namespaced_name("mcp__server__tool")
        assert ns == "mcp"
        assert server == "server"
        assert tool == "tool"

    def test_roundtrip(self) -> None:
        name = namespace_tool_name("search", "my_server")
        ns, server, tool = parse_namespaced_name(name)
        assert ns == DEFAULT_NAMESPACE
        assert server == "my_server"
        assert tool == "search"

    def test_tool_name_with_underscores(self) -> None:
        # Extra __ in tool part still parsed correctly because split(__, 2)
        ns, server, tool = parse_namespaced_name("mcp__srv__my__nested__tool")
        assert ns == "mcp"
        assert server == "srv"
        assert tool == "my__nested__tool"

    def test_invalid_format(self) -> None:
        with pytest.raises(MCPToolError, match="Invalid namespaced"):
            parse_namespaced_name("bad_name")

    def test_invalid_one_separator(self) -> None:
        with pytest.raises(MCPToolError, match="Invalid namespaced"):
            parse_namespaced_name("mcp__only_one")


# ---------------------------------------------------------------------------
# Schema extraction tests
# ---------------------------------------------------------------------------


class TestExtractSchema:
    def test_basic_schema(self) -> None:
        tool = _make_mcp_tool(
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }
        )
        schema = extract_schema(tool)
        assert schema["type"] == "object"
        assert "x" in schema["properties"]
        assert schema["required"] == ["x"]

    def test_empty_schema(self) -> None:
        tool = _make_mcp_tool(input_schema={})
        schema = extract_schema(tool)
        assert isinstance(schema, dict)

    def test_returns_copy(self) -> None:
        original = {"type": "object", "properties": {"a": {"type": "string"}}}
        tool = _make_mcp_tool(input_schema=original)
        schema = extract_schema(tool)
        schema["extra"] = True
        # Original should be unchanged (extract_schema returns a dict copy)
        assert "extra" not in original


# ---------------------------------------------------------------------------
# MCPToolWrapper tests
# ---------------------------------------------------------------------------


class TestMCPToolWrapperInit:
    def test_creation(self) -> None:
        mcp_tool = _make_mcp_tool("search", "Search the web")
        call_fn = AsyncMock()
        wrapper = MCPToolWrapper(mcp_tool, "my_server", call_fn)
        assert wrapper.name == "mcp__my_server__search"
        assert wrapper.description == "Search the web"
        assert wrapper.original_name == "search"
        assert wrapper.server_name == "my_server"
        assert isinstance(wrapper, Tool)

    def test_custom_namespace(self) -> None:
        mcp_tool = _make_mcp_tool("read")
        wrapper = MCPToolWrapper(mcp_tool, "fs", AsyncMock(), namespace="ext")
        assert wrapper.name == "ext__fs__read"

    def test_default_description(self) -> None:
        mcp_tool = MCPTool(
            name="silent",
            description=None,
            inputSchema={"type": "object", "properties": {}},
        )
        wrapper = MCPToolWrapper(mcp_tool, "srv", AsyncMock())
        assert "MCP tool: silent" in wrapper.description

    def test_schema_extraction(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "n": {"type": "integer"},
            },
            "required": ["q"],
        }
        mcp_tool = _make_mcp_tool(input_schema=schema)
        wrapper = MCPToolWrapper(mcp_tool, "srv", AsyncMock())
        assert wrapper.parameters["properties"]["q"]["type"] == "string"
        assert wrapper.parameters["required"] == ["q"]

    def test_to_schema(self) -> None:
        mcp_tool = _make_mcp_tool("search", "Search stuff")
        wrapper = MCPToolWrapper(mcp_tool, "srv", AsyncMock())
        schema = wrapper.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == wrapper.name
        assert schema["function"]["description"] == "Search stuff"


class TestMCPToolWrapperExecute:
    async def test_success(self) -> None:
        call_fn = AsyncMock(return_value=_make_call_result("found it"))
        mcp_tool = _make_mcp_tool("search")
        wrapper = MCPToolWrapper(mcp_tool, "srv", call_fn)

        result = await wrapper.execute(query="test")
        assert result == "found it"
        call_fn.assert_awaited_once_with("search", {"query": "test"})

    async def test_empty_args(self) -> None:
        call_fn = AsyncMock(return_value=_make_call_result("done"))
        mcp_tool = _make_mcp_tool("ping")
        wrapper = MCPToolWrapper(mcp_tool, "srv", call_fn)

        result = await wrapper.execute()
        assert result == "done"
        call_fn.assert_awaited_once_with("ping", None)

    async def test_error_result(self) -> None:
        call_fn = AsyncMock(return_value=_make_call_result("bad input", is_error=True))
        wrapper = MCPToolWrapper(_make_mcp_tool(), "srv", call_fn)

        with pytest.raises(MCPToolError, match="returned error"):
            await wrapper.execute(query="bad")

    async def test_call_fn_exception(self) -> None:
        call_fn = AsyncMock(side_effect=ConnectionError("lost"))
        wrapper = MCPToolWrapper(_make_mcp_tool(), "srv", call_fn)

        with pytest.raises(MCPToolError, match=r"failed.*lost"):
            await wrapper.execute(query="test")

    async def test_multi_content_result(self) -> None:
        result = CallToolResult(
            content=[
                TextContent(type="text", text="line1"),
                TextContent(type="text", text="line2"),
            ],
            isError=False,
        )
        call_fn = AsyncMock(return_value=result)
        wrapper = MCPToolWrapper(_make_mcp_tool(), "srv", call_fn)

        output = await wrapper.execute(query="multi")
        assert output == "line1\nline2"


# ---------------------------------------------------------------------------
# convert_mcp_tools tests
# ---------------------------------------------------------------------------


class TestConvertMCPTools:
    def test_basic_conversion(self) -> None:
        tools = [_make_mcp_tool("a"), _make_mcp_tool("b")]
        call_fn = AsyncMock()
        result = convert_mcp_tools(tools, "srv", call_fn)
        assert len(result) == 2
        assert result[0].name == "mcp__srv__a"
        assert result[1].name == "mcp__srv__b"

    def test_with_filter(self) -> None:
        tools = [_make_mcp_tool("a"), _make_mcp_tool("b"), _make_mcp_tool("c")]
        call_fn = AsyncMock()
        f = MCPToolFilter(include=["a", "c"])
        result = convert_mcp_tools(tools, "srv", call_fn, tool_filter=f)
        assert len(result) == 2
        assert [t.original_name for t in result] == ["a", "c"]

    def test_custom_namespace(self) -> None:
        tools = [_make_mcp_tool("x")]
        result = convert_mcp_tools(tools, "srv", AsyncMock(), namespace="custom")
        assert result[0].name == "custom__srv__x"

    def test_empty_list(self) -> None:
        result = convert_mcp_tools([], "srv", AsyncMock())
        assert result == []

    def test_all_filtered_out(self) -> None:
        tools = [_make_mcp_tool("a")]
        f = MCPToolFilter(exclude=["a"])
        result = convert_mcp_tools(tools, "srv", AsyncMock(), tool_filter=f)
        assert result == []


# ---------------------------------------------------------------------------
# load_tools_from_connection tests
# ---------------------------------------------------------------------------


class TestLoadToolsFromConnection:
    async def test_basic_load(self) -> None:
        conn = MagicMock()
        conn.name = "test-server"
        conn.list_tools = AsyncMock(return_value=[_make_mcp_tool("search"), _make_mcp_tool("read")])
        conn.call_tool = AsyncMock()

        tools = await load_tools_from_connection(conn)
        assert len(tools) == 2
        assert tools[0].original_name == "search"
        assert tools[1].original_name == "read"

    async def test_with_filter(self) -> None:
        conn = MagicMock()
        conn.name = "srv"
        conn.list_tools = AsyncMock(return_value=[_make_mcp_tool("a"), _make_mcp_tool("b")])
        conn.call_tool = AsyncMock()

        tools = await load_tools_from_connection(conn, tool_filter=MCPToolFilter(include=["a"]))
        assert len(tools) == 1
        assert tools[0].original_name == "a"

    async def test_connection_error(self) -> None:
        conn = MagicMock()
        conn.name = "broken"
        conn.list_tools = AsyncMock(side_effect=RuntimeError("disconnected"))

        with pytest.raises(MCPToolError, match="Failed to list tools"):
            await load_tools_from_connection(conn)

    async def test_custom_namespace(self) -> None:
        conn = MagicMock()
        conn.name = "srv"
        conn.list_tools = AsyncMock(return_value=[_make_mcp_tool("x")])
        conn.call_tool = AsyncMock()

        tools = await load_tools_from_connection(conn, namespace="ext")
        assert tools[0].name.startswith("ext__")

    async def test_execution_wired(self) -> None:
        """Verify that loaded tools use the connection's call_tool."""
        conn = MagicMock()
        conn.name = "srv"
        conn.list_tools = AsyncMock(return_value=[_make_mcp_tool("search")])
        conn.call_tool = AsyncMock(return_value=_make_call_result("ok"))

        tools = await load_tools_from_connection(conn)
        result = await tools[0].execute(query="hello")
        assert result == "ok"
        conn.call_tool.assert_awaited_once_with("search", {"query": "hello"})


# ---------------------------------------------------------------------------
# load_tools_from_client tests
# ---------------------------------------------------------------------------


class TestLoadToolsFromClient:
    async def test_single_server(self) -> None:
        conn = MagicMock()
        conn.name = "srv1"
        conn.list_tools = AsyncMock(return_value=[_make_mcp_tool("a")])
        conn.call_tool = AsyncMock()

        client = MagicMock()
        client.server_names = ["srv1"]
        client.connect = AsyncMock(return_value=conn)

        tools = await load_tools_from_client(client)
        assert len(tools) == 1
        assert tools[0].server_name == "srv1"

    async def test_multiple_servers(self) -> None:
        def make_conn(name: str, tool_names: list[str]) -> MagicMock:
            c = MagicMock()
            c.name = name
            c.list_tools = AsyncMock(return_value=[_make_mcp_tool(t) for t in tool_names])
            c.call_tool = AsyncMock()
            return c

        conn1 = make_conn("srv1", ["a", "b"])
        conn2 = make_conn("srv2", ["c"])

        client = MagicMock()
        client.server_names = ["srv1", "srv2"]
        client.connect = AsyncMock(side_effect=[conn1, conn2])

        tools = await load_tools_from_client(client)
        assert len(tools) == 3
        assert tools[0].server_name == "srv1"
        assert tools[2].server_name == "srv2"

    async def test_with_filter(self) -> None:
        conn = MagicMock()
        conn.name = "srv"
        conn.list_tools = AsyncMock(return_value=[_make_mcp_tool("a"), _make_mcp_tool("b")])
        conn.call_tool = AsyncMock()

        client = MagicMock()
        client.server_names = ["srv"]
        client.connect = AsyncMock(return_value=conn)

        tools = await load_tools_from_client(client, tool_filter=MCPToolFilter(exclude=["b"]))
        assert len(tools) == 1
        assert tools[0].original_name == "a"

    async def test_empty_client(self) -> None:
        client = MagicMock()
        client.server_names = []

        tools = await load_tools_from_client(client)
        assert tools == []
