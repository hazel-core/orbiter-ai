# Orbiter

**A modern, modular multi-agent framework for building LLM-powered applications in Python.**

Orbiter is the next-generation rewrite of [AWorld](https://github.com/inclusionAI/AWorld), designed around composability, type safety, and a clean async-first API. Build single agents, multi-agent swarms, and complex orchestration pipelines with minimal boilerplate.

## Features

- **Single Agent class** -- one composable `Agent` with tools, handoffs, hooks, memory, and structured output. No inheritance hierarchies.
- **Type-safe tools** -- the `@tool` decorator auto-generates JSON schemas from function signatures and docstrings.
- **Three execution modes** -- `run()` (async), `run.sync()` (blocking), `run.stream()` (real-time streaming).
- **Multi-agent swarms** -- workflow (sequential pipeline), handoff (agent-driven delegation), and team (lead-worker) modes with a flow DSL: `"researcher >> writer >> reviewer"`.
- **Agent groups** -- `ParallelGroup` and `SerialGroup` for concurrent/sequential sub-pipelines within swarms.
- **Context engine** -- hierarchical state, composable prompt building via neurons, event-driven processors, workspace with artifact versioning.
- **Memory system** -- typed short/long-term memory with SQLite, Postgres, and vector backends.
- **Lifecycle hooks** -- intercept `PRE_LLM_CALL`, `POST_TOOL_CALL`, `START`, `FINISHED`, `ERROR`, and more.
- **Multiple LLM providers** -- OpenAI, Anthropic, Gemini, and Vertex AI built-in. Extensible via `ModelProvider` ABC.
- **MCP support** -- Model Context Protocol client/server for tool interoperability.
- **Structured output** -- constrain agent output to Pydantic models with automatic schema injection.
- **Human-in-the-loop** -- pause agents for human input, confirmation, or review.
- **Config-driven** -- load agents and swarms from YAML configuration files.
- **Tracing** -- OpenTelemetry-based observability with `@traced` decorator.
- **Evaluation** -- rule-based and LLM-as-judge scorers, reflection framework, pass@k.
- **Training** -- trajectory collection, data synthesis, evolution, VeRL/RLHF integration.

## Installation

```bash
# Meta-package -- installs core + all standard extras
pip install orbiter

# Minimal -- agent, tools, runner, swarm only
pip install orbiter-core

# Core + LLM providers
pip install orbiter-core orbiter-models
```

Requires **Python 3.11+**.

## Quick Start

```python
from orbiter import Agent, run, tool


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city.

    Args:
        city: The city to get weather for.
    """
    return f"Sunny, 22°C in {city}."


agent = Agent(
    name="weather-bot",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)

result = run.sync(agent, "What's the weather in Tokyo?")
print(result.output)
```

### Streaming

```python
import asyncio
from orbiter import Agent, run, tool


@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 22°C in {city}."


agent = Agent(name="weather-bot", model="openai:gpt-4o-mini", tools=[get_weather])


async def main():
    async for event in run.stream(agent, "What's the weather in Tokyo?"):
        if event.type == "text":
            print(event.text, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n[Tool: {event.tool_name}]")
    print()


asyncio.run(main())
```

### Multi-Agent Swarm

```python
from orbiter import Agent, Swarm, run

researcher = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions="Research the given topic and provide key facts.",
)

writer = Agent(
    name="writer",
    model="openai:gpt-4o",
    instructions="Write a polished summary from the research notes you receive.",
)

swarm = Swarm(
    agents=[researcher, writer],
    flow="researcher >> writer",
    mode="workflow",
)

result = run.sync(swarm, "Tell me about quantum computing")
print(result.output)
```

### Structured Output

```python
from pydantic import BaseModel
from orbiter import Agent, run


class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    condition: str


agent = Agent(
    name="structured-bot",
    model="openai:gpt-4o-mini",
    instructions="Return weather data in the requested format.",
    output_type=WeatherReport,
)

result = run.sync(agent, "Weather in Paris?")
print(result.output)  # JSON matching WeatherReport schema
```

## Packages

Orbiter is organized as a UV workspace monorepo with 13 focused packages:

| Package | Description |
|---------|-------------|
| [`orbiter-core`](packages/orbiter-core/) | Agent, Tool, Runner, Swarm, Config, Events, Hooks, Registry |
| [`orbiter-models`](packages/orbiter-models/) | LLM providers -- OpenAI, Anthropic, Gemini, Vertex AI |
| [`orbiter-context`](packages/orbiter-context/) | Context engine, neurons, prompt builder, workspace, checkpoints |
| [`orbiter-memory`](packages/orbiter-memory/) | Short/long-term memory, SQLite, Postgres, vector backends |
| [`orbiter-mcp`](packages/orbiter-mcp/) | Model Context Protocol client/server |
| [`orbiter-sandbox`](packages/orbiter-sandbox/) | Local + Kubernetes sandboxed execution |
| [`orbiter-observability`](packages/orbiter-observability/) | Logging, tracing, metrics, health checks, cost tracking |
| [`orbiter-eval`](packages/orbiter-eval/) | Evaluators, scorers, reflection framework |
| [`orbiter-a2a`](packages/orbiter-a2a/) | Agent-to-Agent protocol (server + client) |
| [`orbiter-cli`](packages/orbiter-cli/) | CLI runner, interactive console, batch processing |
| [`orbiter-server`](packages/orbiter-server/) | FastAPI server, session management, WebSocket streaming |
| [`orbiter-train`](packages/orbiter-train/) | Trajectory dataset, data synthesis, VeRL integration |
| [`orbiter`](packages/orbiter/) | Meta-package that installs core + all extras |

## Architecture

```
orbiter (workspace root)
├── packages/
│   ├── orbiter-core        Agent, Tool, Runner, Swarm, Hooks, Events, Config
│   ├── orbiter-models      LLM providers (OpenAI, Anthropic, Gemini, Vertex)
│   ├── orbiter-context     Context engine: state, neurons, processors, workspace
│   ├── orbiter-memory      Short/long-term memory, vector search
│   ├── orbiter-mcp         Model Context Protocol client + server
│   ├── orbiter-sandbox     Sandboxed execution (local, Kubernetes)
│   ├── orbiter-observability Logging, tracing, metrics, observability
│   ├── orbiter-eval        Evaluation, scoring, reflection
│   ├── orbiter-a2a         Agent-to-Agent protocol
│   ├── orbiter-cli         Command-line interface
│   ├── orbiter-server      HTTP server for serving agents
│   ├── orbiter-train       Training: trajectories, synthesis, VeRL
│   └── orbiter             Meta-package (re-exports everything)
├── docs/                   Comprehensive documentation
└── examples/               Runnable demos and benchmarks
```

### Dependency Graph

```
                        orbiter (meta)
                            │
     ┌──────────┬───────────┼───────────┬──────────┐
     │          │           │           │          │
  cli        server      train       a2a        eval
     │          │           │           │          │
     └──────────┴───────────┼───────────┴──────────┘
                            │
     ┌──────────┬───────────┼───────────┬──────────┐
     │          │           │           │          │
  context    memory       mcp       sandbox  observability
     │          │           │           │          │
     └──────────┴───────────┼───────────┴──────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           models         core          (no deps)
              │             │
              └──────── core ◄────────────┘
```

`orbiter-core` has zero heavy dependencies (only `pydantic`). Provider SDKs are isolated in `orbiter-models`.

## Examples

See the [`examples/`](examples/) directory:

- **`quickstart/`** -- define agents, tools, LLM calls, memory, tracing
- **`multi_agent/`** -- workflow, handoff, hybrid swarm, debate, travel planning
- **`advanced/`** -- parallel tasks, HITL, config-driven, serving, CLI
- **`benchmarks/`** -- GAIA, IMO, OSWorld, VisualWebArena, BFCL

## Documentation

Full documentation is in the [`docs/`](docs/) directory:

- **[Getting Started](docs/getting-started/)** -- Installation, quickstart, core concepts, first agent tutorial
- **[Guides](docs/guides/)** -- In-depth guides for every feature (24 guides)
- **[Architecture](docs/architecture/)** -- Design philosophy, dependency graph, execution flow, async patterns
- **[API Reference](docs/reference/)** -- Complete reference for all public APIs (90+ pages)
- **[Contributing](docs/contributing/)** -- Development setup, code style, testing, package structure
- **[Migration Guide](docs/migration-guide.md)** -- Migrating from AWorld to Orbiter

## Development

```bash
# Clone the repository
git clone https://github.com/inclusionAI/AWorld && cd AWorld

# Install UV (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all workspace packages in editable mode
uv sync

# Verify installation
uv run python -c "from orbiter import Agent, run, tool; print('OK')"

# Run all tests (~2,900 tests)
uv run pytest

# Lint
uv run ruff check packages/

# Type-check
uv run pyright packages/orbiter-core/

# Format check
uv run ruff format --check packages/
```

## Environment Variables

```bash
# OpenAI (for "openai:gpt-4o", "openai:gpt-4o-mini", etc.)
export OPENAI_API_KEY="sk-..."

# Anthropic (for "anthropic:claude-sonnet-4-20250514", etc.)
export ANTHROPIC_API_KEY="sk-ant-..."
```

## License

MIT
