"""Orbiter CLI — command-line agent runner.

Entry point for the ``orbiter`` command. Supports agent/swarm execution
from YAML config files with environment variable override, model
selection, verbosity control, and streaming output.

Config file search order (first found wins):
    1. ``--config`` / ``-c`` flag (explicit path)
    2. ``.orbiter.yaml`` in current directory
    3. ``orbiter.config.yaml`` in current directory

Usage::

    orbiter run --config agents.yaml "What is 2+2?"
    orbiter run -m openai:gpt-4o "Hello"
    orbiter --verbose run "Explain Python decorators"
    orbiter start worker --redis-url redis://localhost:6379
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

import typer
from rich.console import Console

# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_NAMES = (".orbiter.yaml", "orbiter.config.yaml")


class CLIError(Exception):
    """Raised for CLI-level errors (config not found, parse failures)."""


def find_config(directory: str | Path | None = None) -> Path | None:
    """Search *directory* (default: cwd) for a config file.

    Returns the first matching path or ``None`` if no config exists.
    """
    base = Path(directory) if directory else Path.cwd()
    for name in _DEFAULT_CONFIG_NAMES:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML config file.

    Delegates to :func:`orbiter.loader.load_yaml` for variable substitution,
    then validates the top-level structure.

    Raises:
        CLIError: If the file doesn't exist or isn't valid YAML dict.
    """
    p = Path(path)
    if not p.is_file():
        raise CLIError(f"Config file not found: {p}")

    from orbiter.loader import LoaderError, load_yaml  # lazy import

    try:
        data = load_yaml(p)
    except LoaderError as exc:
        raise CLIError(f"Invalid config: {exc}") from exc
    return data


def resolve_config(config_path: str | None) -> dict[str, Any] | None:
    """Resolve config from explicit path or auto-discovery.

    Returns:
        Parsed config dict, or ``None`` if no config is available.
    """
    if config_path:
        return load_config(config_path)
    found = find_config()
    if found:
        return load_config(found)
    return None


# ---------------------------------------------------------------------------
# Typer CLI app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="orbiter",
    help="Orbiter — multi-agent framework CLI.",
    no_args_is_help=True,
)

console = Console()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output."),
    ] = False,
) -> None:
    """Orbiter CLI — run agents from the command line."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@app.command()
def run(
    ctx: typer.Context,
    input_text: Annotated[
        str,
        typer.Argument(help="Input text to send to the agent."),
    ],
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to YAML config file."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model string (e.g. openai:gpt-4o)."),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option("--stream", "-s", help="Enable streaming output."),
    ] = False,
) -> None:
    """Run an agent or swarm with the given input."""
    verbose: bool = ctx.obj.get("verbose", False)

    # Resolve config
    cfg = resolve_config(config)
    if verbose and cfg:
        console.print(f"[dim]Loaded config with keys: {list(cfg.keys())}[/dim]")

    # Store parsed state in context for downstream use (loader, console, etc.)
    ctx.obj["config"] = cfg
    ctx.obj["model"] = model
    ctx.obj["stream"] = stream
    ctx.obj["input"] = input_text

    if not cfg:
        console.print("[yellow]No config file found. Use --config or create .orbiter.yaml[/yellow]")
        raise typer.Exit(code=1)

    if verbose:
        console.print(f"[dim]Model: {model or 'auto'}[/dim]")
        console.print(f"[dim]Streaming: {stream}[/dim]")

    console.print(f"[green]Running with input:[/green] {input_text}")


# ---------------------------------------------------------------------------
# Subcommand group: start
# ---------------------------------------------------------------------------

start_app = typer.Typer(
    name="start",
    help="Start long-running services.",
    no_args_is_help=True,
)
app.add_typer(start_app, name="start")


def _mask_redis_url(url: str) -> str:
    """Return a masked version of the Redis URL showing only the host."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "unknown"
        port = parsed.port or 6379
        return f"redis://{host}:{port}/***"
    except Exception:
        return "redis://***"


@start_app.command("worker")
def start_worker(
    redis_url: Annotated[
        str | None,
        typer.Option("--redis-url", help="Redis connection URL (default: ORBITER_REDIS_URL env var)."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Number of concurrent task executions."),
    ] = 1,
    queue: Annotated[
        str,
        typer.Option("--queue", help="Redis Streams queue name."),
    ] = "orbiter:tasks",
    worker_id: Annotated[
        str | None,
        typer.Option("--worker-id", help="Unique worker ID (auto-generated if not set)."),
    ] = None,
) -> None:
    """Start a distributed worker that claims and executes agent tasks."""
    url = redis_url or os.environ.get("ORBITER_REDIS_URL")
    if not url:
        console.print(
            "[red]Error: --redis-url required or set ORBITER_REDIS_URL environment variable.[/red]"
        )
        raise typer.Exit(code=1)

    from orbiter.distributed.worker import Worker  # pyright: ignore[reportMissingImports]

    worker = Worker(
        url,
        worker_id=worker_id,
        concurrency=concurrency,
        queue_name=queue,
    )

    # Print startup banner
    console.print("[bold green]Orbiter Worker Starting[/bold green]")
    console.print(f"  Worker ID:   {worker.worker_id}")
    console.print(f"  Redis URL:   {_mask_redis_url(url)}")
    console.print(f"  Queue:       {queue}")
    console.print(f"  Concurrency: {concurrency}")
    console.print()
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    asyncio.run(worker.start())
