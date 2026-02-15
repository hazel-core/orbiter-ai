# Orbiter

A modern, modular multi-agent framework for building LLM-powered applications.

Orbiter is the next-generation rewrite of [AWorld](https://github.com/inclusionAI/AWorld),
designed around composability, type safety, and a clean async-first API.

## Installation

```bash
pip install orbiter          # meta-package — installs core + all extras
pip install orbiter-core     # minimal — agent, tools, runner, swarm
pip install orbiter-models   # LLM providers (OpenAI, Anthropic)
```

## Quick Start

```python
from orbiter import Agent, run, tool

@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 22 C in {city}."

agent = Agent(
    name="weather-bot",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)

result = run.sync(agent, "What's the weather in Tokyo?")
print(result.output)
```

## Packages

| Package | Description |
|---------|-------------|
| `orbiter-core` | Agent, Tool, Runner, Swarm, Config, Events, Hooks |
| `orbiter-models` | LLM providers (OpenAI, Anthropic) |
| `orbiter-context` | Context engine, neurons, prompt builder, workspace |
| `orbiter-memory` | Short/long-term memory, embeddings, vector search |
| `orbiter-mcp` | Model Context Protocol client/server |
| `orbiter-sandbox` | Local + Kubernetes sandboxed execution |
| `orbiter-trace` | OpenTelemetry tracing, span decorators |
| `orbiter-eval` | Evaluators, scorers, reflection framework |
| `orbiter-a2a` | Agent-to-Agent protocol (server + client) |
| `orbiter-cli` | CLI entry point, agent discovery, interactive console |
| `orbiter-server` | FastAPI server, session management, WebSocket streaming |
| `orbiter-train` | Trajectory dataset, trainers, data synthesis |

## Key Concepts

**Agent** — an LLM-backed unit with tools, instructions, and hooks:

```python
from orbiter import Agent
agent = Agent(name="my-agent", model="openai:gpt-4o", tools=[...])
```

**Tool** — a typed function the agent can call:

```python
from orbiter import tool

@tool
async def search(query: str) -> str:
    """Search the web."""
    return "results..."
```

**Runner** — executes agents with retry, streaming, and handler support:

```python
from orbiter import run

result = await run(agent, "Hello!")            # async
result = run.sync(agent, "Hello!")             # sync wrapper
async for event in run.stream(agent, "Hello!"):  # streaming
    print(event)
```

**Swarm** — multi-agent orchestration (workflow, handoff, team):

```python
from orbiter import Agent, Swarm

researcher = Agent(name="researcher", model="openai:gpt-4o", ...)
writer = Agent(name="writer", model="openai:gpt-4o", ...)

swarm = Swarm(agents=[researcher, writer], mode="workflow")
```

## Examples

See the [`examples/`](examples/) directory for runnable demos:

- `quickstart/` — define agents, tools, LLM calls, memory, tracing
- `multi_agent/` — workflow, handoff, hybrid swarm, debate, travel planning
- `advanced/` — parallel tasks, HITL, config-driven, serving, CLI
- `benchmarks/` — GAIA, IMO, OSWorld, VisualWebArena, BFCL

## Development

```bash
git clone https://github.com/inclusionAI/AWorld && cd AWorld
uv sync                          # install all workspace packages
uv run pytest                    # run tests
uv run ruff check packages/      # lint
uv run pyright packages/orbiter-core/  # type-check
```

## License

MIT
