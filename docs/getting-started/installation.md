# Installation

## Requirements

- **Python 3.11+** -- Orbiter uses modern Python features (`types.UnionType`, `asyncio.TaskGroup`, `ExceptionGroup`) that require Python 3.11 or later.
- **An LLM API key** -- At minimum, you need an API key from OpenAI or Anthropic.

## Install with pip

### Meta-package (recommended)

The `orbiter` meta-package installs orbiter-core plus all standard extras:

```bash
pip install orbiter
```

### Minimal install

If you only need the core agent framework (Agent, Tool, Runner, Swarm) without LLM provider packages:

```bash
pip install orbiter-core
```

### With LLM providers

To use OpenAI and Anthropic models, add the models package:

```bash
pip install orbiter-core orbiter-models
```

### Individual packages

Install only what you need:

```bash
pip install orbiter-core       # Agent, Tool, Runner, Swarm, Config, Events, Hooks
pip install orbiter-models     # OpenAI + Anthropic providers
pip install orbiter-context    # Context engine, neurons, prompt builder
pip install orbiter-memory     # Short/long-term memory, vector search
pip install orbiter-mcp        # Model Context Protocol client/server
pip install orbiter-trace      # OpenTelemetry tracing
pip install orbiter-eval       # Evaluators, scorers, reflection
pip install orbiter-sandbox    # Sandboxed execution environments
pip install orbiter-a2a        # Agent-to-Agent protocol
pip install orbiter-cli        # CLI entry point
pip install orbiter-server     # FastAPI server + WebSocket streaming
pip install orbiter-train      # Trajectory dataset + trainers
```

## Environment Variables

Set API keys for the LLM providers you plan to use:

```bash
# OpenAI (required for "openai:gpt-4o", "openai:gpt-4o-mini", etc.)
export OPENAI_API_KEY="sk-..."

# Anthropic (required for "anthropic:claude-sonnet-4-20250514", etc.)
export ANTHROPIC_API_KEY="sk-ant-..."
```

You can also pass API keys programmatically when constructing a provider:

```python
from orbiter.models.provider import get_provider

provider = get_provider("openai:gpt-4o", api_key="sk-...")
```

Or set them in a `.env` file and load with your preferred method (e.g., `python-dotenv`).

## Development Setup (UV Workspace)

If you want to contribute to Orbiter or work with the full monorepo:

```bash
# Clone the repository
git clone https://github.com/inclusionAI/AWorld && cd AWorld

# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all workspace packages (installs all 13 packages in editable mode)
uv sync

# Verify the installation
uv run python -c "from orbiter import Agent, run, tool; print('OK')"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific package
uv run pytest packages/orbiter-core/tests/
uv run pytest packages/orbiter-models/tests/

# Run with verbose output
uv run pytest -v
```

### Linting and Type Checking

```bash
# Lint with ruff
uv run ruff check packages/

# Type-check with pyright
uv run pyright packages/orbiter-core/
uv run pyright packages/orbiter-models/
```

## Verify Your Installation

Run this minimal script to confirm everything is working:

```python
from orbiter import Agent, tool

@tool
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

agent = Agent(
    name="greeter",
    model="openai:gpt-4o-mini",
    instructions="You are a friendly greeter.",
    tools=[hello],
)

print(agent.describe())
# {'name': 'greeter', 'model': 'openai:gpt-4o-mini', 'tools': ['hello'],
#  'handoffs': [], 'max_steps': 10, 'output_type': None}
```

This script only constructs the agent and does not make any LLM calls, so it works without an API key. If this runs without errors, your installation is correct.

## Next Steps

Continue to the [Quickstart](quickstart.md) to build and run your first agent.
