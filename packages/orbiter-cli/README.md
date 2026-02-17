# orbiter-cli

Command-line interface for the [Orbiter](../../README.md) multi-agent framework.

## Installation

```bash
pip install orbiter-cli
```

Requires Python 3.11+, `orbiter-core`, `orbiter-models`, `typer>=0.12`, and `rich>=13.0`.

## Usage

After installation, the `orbiter` command is available:

```bash
# Run an agent from a YAML config file
orbiter run agents.yaml

# Interactive console mode
orbiter console

# Run in batch mode
orbiter batch input.jsonl --output results.jsonl

# Discover agents in the current directory
orbiter list
```

## What's Included

- **Agent runner** -- load and run agents from YAML configuration or Python modules.
- **Interactive console** -- rich terminal UI for conversing with agents.
- **Batch processing** -- process input files with agents at scale.
- **Agent discovery** -- scan directories for agent definitions.
- **Plugin system** -- extend the CLI with custom commands.

## Documentation

- [CLI Guide](../../docs/guides/cli.md)
- [API Reference](../../docs/reference/cli/)
