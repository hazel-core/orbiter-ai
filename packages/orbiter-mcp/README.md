# orbiter-mcp

[Model Context Protocol](https://modelcontextprotocol.io/) (MCP) client and server for the [Orbiter](../../README.md) multi-agent framework.

## Installation

```bash
pip install orbiter-mcp
```

Requires Python 3.11+, `orbiter-core`, and `mcp>=1.0`.

## What's Included

- **MCP Client** -- connect to MCP servers and load their tools as Orbiter tools.
- **MCP Server** -- expose Orbiter tools as an MCP server using decorators.
- **Multi-server management** -- connect to multiple MCP servers simultaneously.
- **Tool bridge** -- automatic conversion between MCP tool schemas and Orbiter tool schemas.

## Quick Example

```python
from orbiter import Agent
from orbiter.mcp import MCPClient

# Connect to an MCP server and load its tools
client = MCPClient("npx @anthropic/mcp-server-filesystem /tmp")
tools = await client.get_tools()

agent = Agent(
    name="file-agent",
    model="openai:gpt-4o",
    tools=tools,
)
```

## Documentation

- [MCP Guide](../../docs/guides/mcp.md)
- [API Reference](../../docs/reference/mcp/)
