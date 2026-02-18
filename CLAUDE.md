# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Orbiter

Orbiter is a modular multi-agent framework for building LLM-powered applications in Python. It's a UV workspace monorepo with 15 packages. Requires Python 3.11+.

## Common Commands

```bash
# Install all workspace packages (editable mode)
uv sync

# Run all tests (~2,900 tests, asyncio_mode=auto)
uv run pytest

# Run tests for a single package
uv run pytest packages/orbiter-core/tests/

# Run a single test file
uv run pytest packages/orbiter-core/tests/test_agent.py

# Run a single test
uv run pytest packages/orbiter-core/tests/test_agent.py::test_function_name

# Lint (with auto-fix)
uv run ruff check packages/ --fix

# Format check
uv run ruff format --check packages/

# Type-check a package
uv run pyright packages/orbiter-core/

# Verify installation
uv run python -c "from orbiter import Agent, run, tool; print('OK')"
```

### orbiter-web (dual Node+Python package)

```bash
cd packages/orbiter-web

# Install frontend deps
npm install

# Dev server (runs Astro + FastAPI concurrently)
npm run dev

# Astro typecheck
npx astro check

# Run backend only
uv run uvicorn orbiter_web.app:app --reload
```

## Architecture

UV workspace monorepo. Packages live in `packages/`. The dependency graph flows upward from `orbiter-core`:

```
orbiter-core (foundation, only depends on pydantic)
    ↑
orbiter-models (OpenAI, Anthropic, Gemini, Vertex AI providers)
    ↑
orbiter-context, orbiter-memory, orbiter-mcp, orbiter-sandbox, orbiter-observability
    ↑
orbiter-cli, orbiter-server, orbiter-eval, orbiter-a2a, orbiter-train, orbiter-web
    ↑
orbiter (meta-package, re-exports everything)
```

### Key Packages

- **orbiter-core** (`packages/orbiter-core/src/orbiter/`): `Agent`, `Tool`, `@tool` decorator, `run`/`run.sync`/`run.stream`, `Swarm`, hooks, events, config, registry. The `_internal/` subpackage has message building, output parsing, call execution, state machine, and graph algorithms.
- **orbiter-models** (`packages/orbiter-models/`): LLM provider implementations. Provider SDKs are isolated here — core has zero heavy deps.
- **orbiter-web** (`packages/orbiter-web/`): Full platform UI. Hybrid package — Astro 5.x frontend (`src/pages/`, `src/islands/`) + FastAPI backend (`src/orbiter_web/`). Has its own `package.json` AND `pyproject.toml`.

### orbiter-web Backend Structure

- `app.py` — FastAPI app entry point, middleware, route registration
- `config.py` — Settings dataclass (env vars: `ORBITER_DATABASE_URL`, `ORBITER_SECRET_KEY`, `ORBITER_DEBUG`)
- `database.py` — `get_db()` async context manager, WAL mode, foreign keys
- `engine.py` — Workflow execution engine (topological sort, node execution, retry)
- `migrations/` — Sequential SQL files, run automatically on startup via lifespan
- `routes/` — 30+ APIRouter modules, all under `/api/v1/` prefix
- `services/` — Business logic layer (agent runtime, sandbox, scheduler, memory)
- `middleware/` — CSRF, rate limiting, security headers, API version redirect

### orbiter-web Frontend Structure

- Astro 5.x pages in `src/pages/`, layouts in `src/layouts/`
- React islands in `src/islands/` (e.g., ReactFlow canvas)
- Tailwind CSS v4 via `@tailwindcss/vite`
- `cn()` utility at `src/utils/merge.ts` for class merging

## Code Conventions

- **Ruff**: line-length 100, rules `E,F,I,N,W,UP,B,SIM,RUF`, ignore `E501`. Use `datetime.UTC` not `timezone.utc`.
- **Pyright**: basic mode, Python 3.11 target.
- **Async-first**: all core APIs are async. Tests use `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed).
- **Pydantic v2**: for all schemas and validation.
- **Test file names must be unique** across all packages (pytest importlib mode).
- **Tests use MockProvider** — never make real API calls.
- **Model strings**: format `"provider:model"` (e.g., `"openai:gpt-4o-mini"`).
- **FastAPI Depends()**: use `# noqa: B008` for ruff on function defaults.
- **CSRF**: auto-injected via fetch monkey-patch in PageLayout — no manual header needed in frontend.
- **API routes**: define static paths (`/search`) before param routes (`/{id}`) to prevent FastAPI mismatching.

## Adding a New Package to the Workspace

1. Create `packages/<name>/` with `pyproject.toml` and `src/` layout
2. Update root `pyproject.toml`: add to `[tool.uv.workspace].members`, `[dependency-groups].dev`, and `[tool.uv.sources]`
3. Run `uv sync`

## Important File Locations

- Root config: `pyproject.toml` (workspace definition, ruff, pyright, pytest config)
- Public API exports: `packages/orbiter-core/src/orbiter/__init__.py`
- Provider resolution: `packages/orbiter-models/`
- Web app entry: `packages/orbiter-web/src/orbiter_web/app.py`
- DB migrations: `packages/orbiter-web/src/orbiter_web/migrations/`
- Handle types (keep in sync): `packages/orbiter-web/src/islands/Canvas/handleTypes.ts` ↔ `routes/tools.py` (`_NODE_HANDLE_MAP`)
