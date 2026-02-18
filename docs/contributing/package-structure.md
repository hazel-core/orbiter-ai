# Package Structure

Orbiter is a UV workspace monorepo with 13 packages. This document describes the conventions for package layout, namespace packages, and how to add new packages.

## Standard Package Layout

Every package follows this structure:

```
packages/orbiter-<name>/
+-- pyproject.toml
+-- src/
|   +-- orbiter/
|       +-- __init__.py          # extend_path + re-exports
|       +-- <name>/              # (for sub-namespace packages)
|       |   +-- __init__.py      # public API
|       |   +-- module.py
|       |   +-- _internal/       # private implementation details
|       |       +-- __init__.py
|       |       +-- helper.py
|       +-- module.py            # (for top-level namespace)
+-- tests/
    +-- __init__.py
    +-- test_module.py
```

### orbiter-core (top-level namespace)

`orbiter-core` contributes modules directly to the `orbiter` namespace:

```
packages/orbiter-core/
+-- src/
|   +-- orbiter/
|       +-- __init__.py      # extend_path, re-exports Agent, Swarm, Tool, tool, run
|       +-- types.py         # Message types, RunResult, StreamEvent
|       +-- config.py        # AgentConfig, ModelConfig, etc.
|       +-- registry.py      # Registry[T]
|       +-- events.py        # EventBus
|       +-- hooks.py         # Hook, HookPoint, HookManager
|       +-- tool.py          # Tool ABC, @tool decorator, FunctionTool
|       +-- agent.py         # Agent class
|       +-- swarm.py         # Swarm class
|       +-- runner.py        # run(), run.sync(), run.stream()
|       +-- _internal/
|           +-- __init__.py
|           +-- message_builder.py
|           +-- output_parser.py
|           +-- call_runner.py
|           +-- state.py
|           +-- graph.py
+-- tests/
    +-- test_types.py
    +-- test_config.py
    +-- test_agent.py
    +-- ...
```

### Sub-namespace packages (orbiter-models, orbiter-context, etc.)

Other packages contribute to sub-namespaces like `orbiter.models`, `orbiter.context`:

```
packages/orbiter-models/
+-- src/
|   +-- orbiter/
|       +-- __init__.py          # extend_path (namespace extension)
|       +-- models/
|           +-- __init__.py      # re-exports: ModelProvider, get_provider, etc.
|           +-- types.py         # ModelError, ModelResponse, StreamChunk
|           +-- provider.py      # ModelProvider ABC, model_registry
|           +-- openai.py        # OpenAIProvider
|           +-- anthropic.py     # AnthropicProvider
+-- tests/
    +-- test_model_types.py      # Note: prefixed to avoid collision with orbiter-core
    +-- ...
```

## Namespace Packages

Orbiter uses `pkgutil.extend_path()` to create namespace packages. This allows multiple PyPI packages to contribute to the same `orbiter` import path.

### How It Works

Each package's `orbiter/__init__.py` calls `extend_path()`:

```python
"""Orbiter Core: Agent, Tool, Runner, Config, Events, Hooks, Swarm."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

# Public API re-exports
from orbiter.agent import Agent
from orbiter.runner import run
from orbiter.swarm import Swarm
from orbiter.tool import FunctionTool, Tool, tool
```

When Python imports `orbiter`, it finds the `orbiter/` directory from `orbiter-core`. The `extend_path()` call tells Python to also search other installed packages for `orbiter/` directories. This means `orbiter-models` can add `orbiter/models/` and it becomes importable as `from orbiter.models import ...`.

### Important Details

- **Every package needs `__init__.py`** in its `orbiter/` directory with `extend_path()`
- **Re-exports only in `__init__.py`** -- no logic, no complex imports
- **`__all__` is recommended** when the export list might be ambiguous
- **The meta-package** (`packages/orbiter/`) uses a `_orbiter_meta` dummy package for hatchling compatibility

## `__init__.py` Pattern

```python
"""<Package description>."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

# Public API — these are what users import
from orbiter.agent import Agent
from orbiter.runner import run
# ...

__all__ = [
    "Agent",
    "run",
    # ...
]
```

### Rules

- `__init__.py` is the **public API surface** -- only export what users need
- Use `__all__` if the export list is ambiguous
- **Never put logic in `__init__.py`** -- only imports and re-exports
- Subpackages (`orbiter.models`, `orbiter.context`, etc.) follow the same pattern

## pyproject.toml Template

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "orbiter-<name>"
version = "0.1.0"
description = "<description>"
requires-python = ">=3.11"
dependencies = [
    "orbiter-core>=0.1.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/orbiter"]

[tool.uv.sources]
orbiter-core = { workspace = true }
```

### Key Points

- **Build system:** hatchling (consistent across all packages)
- **Python version:** 3.11+ (required for `asyncio.TaskGroup`, `StrEnum`)
- **Workspace sources:** Use `{ workspace = true }` for workspace dependencies so UV resolves them locally
- **Wheel packages:** Point to `src/orbiter` so the namespace package structure is preserved

## Naming Conventions

| Thing | Convention | Example |
|-------|-----------|---------|
| Package directory | `orbiter-<name>` | `orbiter-core`, `orbiter-models` |
| PyPI package name | `orbiter-<name>` | `pip install orbiter-models` |
| Import path | `orbiter.<name>` | `from orbiter.models import ...` |
| Module files | `snake_case.py` | `message_builder.py`, `output_parser.py` |
| Internal modules | `_internal/<name>.py` | `_internal/call_runner.py` |

## Dependency Rules

1. **`orbiter-core` has ZERO heavy dependencies** -- only `pydantic`
2. **Provider SDKs** live in `orbiter-models` only (e.g., `openai`, `anthropic`)
3. **Optional heavy deps** declared as extras (e.g., `orbiter-memory[qdrant]`)
4. **No lateral dependencies** between same-level packages (e.g., `orbiter-models` does not depend on `orbiter-memory`)
5. **`_internal/` modules are never imported from outside their package**

## Adding a New Package

1. Create the directory structure:
   ```bash
   mkdir -p packages/orbiter-foo/src/orbiter/foo
   mkdir -p packages/orbiter-foo/tests
   ```

2. Create `packages/orbiter-foo/pyproject.toml` (see template above)

3. Create `packages/orbiter-foo/src/orbiter/__init__.py`:
   ```python
   from pkgutil import extend_path
   __path__ = extend_path(__path__, __name__)
   ```

4. Create `packages/orbiter-foo/src/orbiter/foo/__init__.py`:
   ```python
   """Orbiter Foo: <description>."""
   # Public API re-exports
   ```

5. Create `packages/orbiter-foo/tests/__init__.py` (empty)

6. Add to workspace root `pyproject.toml` members list

7. Run `uv sync`

## Test File Uniqueness

Test file names must be **unique across all packages**. pytest collects `tests/` from multiple packages into one module namespace.

```
# BAD -- collision between orbiter-core and orbiter-models
packages/orbiter-core/tests/test_types.py
packages/orbiter-models/tests/test_types.py     # NAME COLLISION

# GOOD -- prefixed names
packages/orbiter-core/tests/test_types.py
packages/orbiter-models/tests/test_model_types.py
```
