# orbiter

Meta-package for the [Orbiter](../../README.md) multi-agent framework. Installs `orbiter-core` plus all standard extras in a single command.

## Installation

```bash
pip install orbiter
```

This installs:

- `orbiter-core` -- Agent, Tool, Runner, Swarm, Config, Events, Hooks
- `orbiter-models` -- LLM providers (OpenAI, Anthropic, Gemini)
- `orbiter-context` -- Context engine, neurons, prompt builder
- `orbiter-memory` -- Short/long-term memory, vector search
- `orbiter-mcp` -- Model Context Protocol client/server
- `orbiter-sandbox` -- Sandboxed execution environments
- `orbiter-observability` -- Logging, tracing, metrics, and health checks
- `orbiter-eval` -- Evaluation and scoring
- `orbiter-a2a` -- Agent-to-Agent protocol

CLI (`orbiter-cli`), server (`orbiter-server`), and training (`orbiter-train`) are installed separately.

## Quick Start

```python
from orbiter import Agent, run, tool


@tool
async def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


agent = Agent(
    name="greeter",
    model="openai:gpt-4o-mini",
    instructions="You are a friendly greeter.",
    tools=[greet],
)

result = run.sync(agent, "Say hi to Alice")
print(result.output)
```

## Documentation

See the full [Orbiter documentation](../../docs/).
