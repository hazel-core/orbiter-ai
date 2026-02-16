"""Orbiter CLI: Command-line agent runner."""

from orbiter_cli.batch import (
    BatchError,
    BatchItem,
    BatchResult,
    InputFormat,
    ItemResult,
    load_batch_items,
    results_to_csv,
    results_to_jsonl,
)
from orbiter_cli.console import InteractiveConsole, format_agents_table, parse_command
from orbiter_cli.executor import ExecutionResult, ExecutorError, LocalExecutor
from orbiter_cli.loader import (
    AgentLoadError,
    discover_agent_files,
    load_markdown_agent,
    load_python_agent,
    load_yaml_agents,
    scan_directory,
    validate_agent,
)
from orbiter_cli.main import CLIError, app, find_config, load_config, resolve_config
from orbiter_cli.plugins import PluginError, PluginHook, PluginManager, PluginSpec

__all__ = [
    "AgentLoadError",
    "BatchError",
    "BatchItem",
    "BatchResult",
    "CLIError",
    "ExecutionResult",
    "ExecutorError",
    "InputFormat",
    "InteractiveConsole",
    "ItemResult",
    "LocalExecutor",
    "PluginError",
    "PluginHook",
    "PluginManager",
    "PluginSpec",
    "app",
    "discover_agent_files",
    "find_config",
    "format_agents_table",
    "load_batch_items",
    "load_config",
    "load_markdown_agent",
    "load_python_agent",
    "load_yaml_agents",
    "parse_command",
    "resolve_config",
    "results_to_csv",
    "results_to_jsonl",
    "scan_directory",
    "validate_agent",
]
