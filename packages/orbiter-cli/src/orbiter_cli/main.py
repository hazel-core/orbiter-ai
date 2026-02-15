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
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

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
