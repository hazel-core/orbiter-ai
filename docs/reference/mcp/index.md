# orbiter.mcp

Model Context Protocol client, server, and tool integration for Orbiter.

## Installation

```bash
pip install orbiter-mcp
```

## Module path

```python
import orbiter.mcp
```

## Package exports

| Export | Module | Description |
|---|---|---|
| `MCPClient` | `orbiter.mcp.client` | High-level client managing multiple MCP server connections |
| `MCPClientError` | `orbiter.mcp.client` | Error raised by MCP client operations |
| `MCPServerConfig` | `orbiter.mcp.client` | Configuration for an MCP server connection |
| `MCPServerConnection` | `orbiter.mcp.client` | A live connection to an MCP server |
| `MCPTransport` | `orbiter.mcp.client` | Enum of transport types (stdio, sse, streamable_http) |
| `MCPExecutionError` | `orbiter.mcp.execution` | Error raised during MCP tool execution with retries |
| `load_mcp_client` | `orbiter.mcp.execution` | Create an MCPClient from an mcp.json config file |
| `load_mcp_config` | `orbiter.mcp.execution` | Load MCP server configurations from an mcp.json file |
| `substitute_env_vars` | `orbiter.mcp.execution` | Replace `${VAR_NAME}` placeholders with environment variable values |
| `MCPServerError` | `orbiter.mcp.server` | Error raised by MCP server operations |
| `MCPServerRegistry` | `orbiter.mcp.server` | Singleton registry for @mcp_server-decorated classes |
| `mcp_server` | `orbiter.mcp.server` | Class decorator that converts a Python class into an MCP server |
| `MCPToolError` | `orbiter.mcp.tools` | Error raised during MCP tool operations |
| `MCPToolFilter` | `orbiter.mcp.tools` | Filter for including/excluding MCP tools by name |
| `MCPToolWrapper` | `orbiter.mcp.tools` | An Orbiter Tool wrapping an MCP tool for execution |
| `convert_mcp_tools` | `orbiter.mcp.tools` | Convert a list of MCP tools to Orbiter MCPToolWrapper instances |
| `extract_schema` | `orbiter.mcp.tools` | Extract JSON Schema parameters from an MCP tool |
| `namespace_tool_name` | `orbiter.mcp.tools` | Create a namespaced tool name: `{namespace}__{server}__{tool}` |
| `parse_namespaced_name` | `orbiter.mcp.tools` | Parse a namespaced tool name back into (namespace, server, tool) |

## Submodules

- [orbiter.mcp.client](client.md) -- Client classes and transport configuration
- [orbiter.mcp.execution](execution.md) -- Config loading, env var substitution, retry logic
- [orbiter.mcp.server](server.md) -- `@mcp_server` decorator and server registry
- [orbiter.mcp.tools](tools.md) -- Tool wrappers, filters, schema extraction, namespacing
