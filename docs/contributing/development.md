# Development Setup

This guide covers setting up your development environment for contributing to Orbiter.

## Prerequisites

- **Python 3.11+** (required for `asyncio.TaskGroup` and `StrEnum`)
- **UV** -- Orbiter uses UV for workspace management and dependency resolution

Install UV if you do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
# Clone the repository
git clone https://github.com/anthropics/orbiter.git
cd orbiter

# Install all workspace packages in editable mode
uv sync
```

`uv sync` installs all 13 packages from `packages/` in editable mode, plus all development dependencies. This means changes to source files take effect immediately without reinstalling.

## Workspace Structure

The repository is a UV workspace monorepo:

```
orbiter/
+-- pyproject.toml              # Workspace root (NOT a package)
+-- packages/
    +-- orbiter-core/           # Core types, Agent, Tool, Swarm, Runner
    +-- orbiter-models/         # LLM provider abstractions
    +-- orbiter-context/        # Context engine
    +-- orbiter-memory/         # Memory backends
    +-- orbiter-mcp/            # MCP client/server
    +-- orbiter-sandbox/        # Sandboxed execution
    +-- orbiter-trace/          # Tracing
    +-- orbiter-eval/           # Evaluation
    +-- orbiter-a2a/            # Agent-to-Agent protocol
    +-- orbiter-cli/            # CLI
    +-- orbiter-server/         # HTTP server
    +-- orbiter-train/          # Training
    +-- orbiter/                # Meta-package
```

The root `pyproject.toml` defines the workspace but is **not** a package itself. It configures shared dev dependencies, pytest settings, and ruff/pyright rules.

## Running Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific package
uv run pytest packages/orbiter-core/tests/
uv run pytest packages/orbiter-models/tests/

# Run a specific test file
uv run pytest packages/orbiter-core/tests/test_types.py

# Run a specific test function
uv run pytest packages/orbiter-core/tests/test_agent.py::test_agent_calls_tool

# Run with verbose output
uv run pytest -v

# Run with print output visible
uv run pytest -s
```

### pytest Configuration

The root `pyproject.toml` configures pytest:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

This means:
- **All `async def test_*` functions run automatically** -- no need for `@pytest.mark.asyncio`
- Tests are collected from all `packages/*/tests/` directories
- `--import-mode=importlib` is used to handle multiple `tests/` packages

## Linting

Orbiter uses **ruff** for linting and import sorting:

```bash
# Check for lint errors
uv run ruff check packages/

# Auto-fix what can be auto-fixed
uv run ruff check packages/ --fix

# Format code
uv run ruff format packages/
```

Key ruff rules:
- Line length: 100 characters
- Import sorting: `isort`-compatible (stdlib, third-party, local)
- Double quotes for strings

## Type Checking

Orbiter uses **pyright** in strict mode:

```bash
# Type check orbiter-core
uv run pyright packages/orbiter-core/

# Type check orbiter-models
uv run pyright packages/orbiter-models/

# Type check a specific file
uv run pyright packages/orbiter-core/src/orbiter/types.py
```

### Known pyright Limitations

Pyright cannot resolve cross-namespace-package imports from editable installs (`.pth`-based). For example, importing `orbiter.models.types` from a test file in `orbiter-core` will show a `reportMissingImports` error even though the import works at runtime.

Workaround: add `# pyright: ignore[reportMissingImports]` on the specific import line:

```python
from orbiter.models.types import ModelError  # pyright: ignore[reportMissingImports]
```

Only use this in **test files** that cross package boundaries. Source files should not need this.

## Common Development Tasks

### Adding a new module to an existing package

1. Create the module file in `packages/orbiter-<pkg>/src/orbiter/<pkg>/`
2. Add public exports to `__init__.py`
3. Create a test file in `packages/orbiter-<pkg>/tests/`
4. Run tests and type checker

### Adding a new dependency to a package

1. Edit `packages/orbiter-<pkg>/pyproject.toml`
2. Add the dependency to `[project.dependencies]`
3. Run `uv sync` to update the lockfile

### Adding a workspace dependency

If package A depends on package B within the workspace:

1. Add B to A's `[project.dependencies]`: `orbiter-core>=0.1.0`
2. Add B to A's `[tool.uv.sources]`:
   ```toml
   [tool.uv.sources]
   orbiter-core = { workspace = true }
   ```
3. Run `uv sync`

## Environment Variables

For running integration tests (optional):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for provider integration tests |
| `ANTHROPIC_API_KEY` | Anthropic API key for provider integration tests |

Unit tests do not require any API keys -- they use mock providers.
