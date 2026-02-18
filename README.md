<div align="center">

# Orbiter

### A modern, modular multi-agent framework for Python

Build single agents, multi-agent swarms, and complex orchestration pipelines — with minimal boilerplate.

[Getting Started](docs/getting-started/) &nbsp;&bull;&nbsp; [Guides](docs/guides/) &nbsp;&bull;&nbsp; [API Reference](docs/reference/) &nbsp;&bull;&nbsp; [Examples](examples/)

---

</div>

## Why Orbiter

Orbiter is the next-generation rewrite of [AWorld](https://github.com/inclusionAI/AWorld), designed around **composability**, **type safety**, and a clean **async-first** API.

One `Agent` class. Three execution modes. Zero inheritance hierarchies.

```python
from orbiter import Agent, run, tool

@tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
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

## Highlights

<table>
<tr>
<td width="50%">

**Core**
- Single composable `Agent` with tools, handoffs, hooks, memory, and structured output
- `@tool` decorator auto-generates JSON schemas from signatures and docstrings
- Three modes: `run()` async, `run.sync()` blocking, `run.stream()` real-time
- Lifecycle hooks: `PRE_LLM_CALL`, `POST_TOOL_CALL`, `START`, `FINISHED`, `ERROR`
- Config-driven: load agents and swarms from YAML

</td>
<td width="50%">

**Multi-Agent**
- Workflow (sequential), handoff (agent-driven), and team (lead-worker) swarm modes
- Flow DSL: `"researcher >> writer >> reviewer"`
- `ParallelGroup` and `SerialGroup` for concurrent/sequential sub-pipelines
- Agent-to-Agent protocol (A2A) for network-based delegation
- Distributed execution with Redis task queue and Temporal workflows

</td>
</tr>
<tr>
<td width="50%">

**Intelligence**
- Context engine with hierarchical state, neurons, and workspace
- Short/long-term memory with SQLite, Postgres, and vector backends
- Structured output constrained to Pydantic models
- Skills system for dynamic, reusable capability packages
- Human-in-the-loop: pause for input, confirmation, or review

</td>
<td width="50%">

**Platform**
- LLM providers: OpenAI, Anthropic, Gemini, Vertex AI — extensible via `ModelProvider` ABC
- MCP client/server for tool interoperability
- OpenTelemetry tracing with `@traced` decorator
- Evaluation: rule-based and LLM-as-judge scorers, reflection, pass@k
- Training: trajectory collection, data synthesis, VeRL/RLHF integration

</td>
</tr>
</table>

**Orbiter Web** — Full-featured agent platform with visual workflow editor, real-time playground, knowledge bases, scheduling, and team management. Built with Astro 5 + React 19 + FastAPI.

## Installation

```bash
# Meta-package — installs core + all standard extras
pip install orbiter

# Minimal — agent, tools, runner, swarm only
pip install orbiter-core

# Core + LLM providers
pip install orbiter-core orbiter-models

# Distributed execution
pip install orbiter-distributed
```

> Requires **Python 3.11+**

## Quick Start

### Streaming

Rich streaming events provide real-time visibility into agent execution:

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

Enable detailed events for full visibility:

```python
async for event in run.stream(agent, "...", detailed=True):
    # TextEvent, ToolCallEvent, ToolResultEvent, StepEvent, ErrorEvent, and more
    ...
```

### Multi-Agent Swarm

Chain agents with the flow DSL:

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

Constrain agent output to Pydantic models:

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
print(result.output)  # WeatherReport(city='Paris', temperature_celsius=18.0, condition='Partly cloudy')
```

## Packages

Orbiter is organized as a UV workspace monorepo with 15 focused packages:

| Package | Description |
|:--------|:------------|
| [`orbiter-core`](packages/orbiter-core/) | Agent, Tool, Runner, Swarm, Config, Events, Hooks, Registry |
| [`orbiter-models`](packages/orbiter-models/) | LLM providers — OpenAI, Anthropic, Gemini, Vertex AI |
| [`orbiter-context`](packages/orbiter-context/) | Context engine, neurons, prompt builder, workspace, checkpoints |
| [`orbiter-memory`](packages/orbiter-memory/) | Short/long-term memory, SQLite, Postgres, vector backends |
| [`orbiter-mcp`](packages/orbiter-mcp/) | Model Context Protocol client/server |
| [`orbiter-sandbox`](packages/orbiter-sandbox/) | Local + Kubernetes sandboxed execution |
| [`orbiter-observability`](packages/orbiter-observability/) | Logging, tracing, metrics, health checks, cost tracking |
| [`orbiter-distributed`](packages/orbiter-distributed/) | Redis task queue, workers, Temporal workflows, event streaming |
| [`orbiter-eval`](packages/orbiter-eval/) | Evaluators, scorers, reflection framework |
| [`orbiter-a2a`](packages/orbiter-a2a/) | Agent-to-Agent protocol (server + client) |
| [`orbiter-cli`](packages/orbiter-cli/) | CLI runner, interactive console, batch processing |
| [`orbiter-server`](packages/orbiter-server/) | FastAPI server, session management, WebSocket streaming |
| [`orbiter-train`](packages/orbiter-train/) | Trajectory dataset, data synthesis, VeRL integration |
| [`orbiter-web`](packages/orbiter-web/) | Full platform UI — visual workflows, playground, knowledge bases |
| [`orbiter`](packages/orbiter/) | Meta-package that installs core + all extras |

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
            models         core        distributed
               │             │             │
               └──────── core ◄────────────┘
```

`orbiter-core` has zero heavy dependencies (only `pydantic`). Provider SDKs are isolated in `orbiter-models`.

## Examples

See the [`examples/`](examples/) directory:

| Directory | What's Inside |
|:----------|:--------------|
| `quickstart/` | Agents, tools, LLM calls, memory, tracing, config-driven, MCP |
| `multi_agent/` | Workflow, handoff, debate, deep research, master-worker, travel planning |
| `advanced/` | Parallel tasks, HITL, skills, web deployment |
| `distributed/` | Redis workers, SSE streaming, multi-agent distributed execution |
| `benchmarks/` | GAIA, IMO, OSWorld, VisualWebArena, BFCL, XBench |

## Documentation

Full documentation is in the [`docs/`](docs/) directory:

| Section | Description |
|:--------|:------------|
| [Getting Started](docs/getting-started/) | Installation, quickstart, core concepts, first agent tutorial |
| [Guides](docs/guides/) | 28 in-depth guides covering every feature |
| [Architecture](docs/architecture/) | Design philosophy, dependency graph, execution flow, async patterns |
| [API Reference](docs/reference/) | Complete reference for all public APIs |
| [Contributing](docs/contributing/) | Development setup, code style, testing, package structure |
| [Migration Guide](docs/migration-guide.md) | Migrating from AWorld to Orbiter |

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

# Run all tests
uv run pytest

# Lint + format
uv run ruff check packages/
uv run ruff format --check packages/

# Type-check
uv run pyright packages/orbiter-core/
```

### Orbiter Web (Platform UI)

```bash
cd packages/orbiter-web
npm install
npm run dev          # Runs Astro frontend + FastAPI backend concurrently
```

## Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Vertex AI (uses service account or ADC)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
# OR
export VERTEX_PROJECT="my-project"
export VERTEX_LOCATION="us-central1"
```

## License

MIT
