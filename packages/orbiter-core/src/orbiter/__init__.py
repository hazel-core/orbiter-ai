"""Orbiter Core: Agent, Tool, Runner, Config, Events, Hooks, Swarm."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

from orbiter._internal.agent_group import ParallelGroup, SerialGroup
from orbiter._internal.nested import SwarmNode
from orbiter.agent import Agent
from orbiter.observability.logging import (  # pyright: ignore[reportMissingImports]
    configure_logging as configure,
)
from orbiter.observability.logging import (  # pyright: ignore[reportMissingImports]
    get_logger,
)
from orbiter.runner import run
from orbiter.swarm import Swarm
from orbiter.tool import FunctionTool, Tool, tool
from orbiter.types import (
    AudioBlock,
    ContentBlock,
    DocumentBlock,
    ImageDataBlock,
    ImageURLBlock,
    MessageContent,
    TextBlock,
    VideoBlock,
)

__all__ = [
    "Agent",
    "AudioBlock",
    "ContentBlock",
    "DocumentBlock",
    "FunctionTool",
    "ImageDataBlock",
    "ImageURLBlock",
    "MessageContent",
    "ParallelGroup",
    "SerialGroup",
    "Swarm",
    "SwarmNode",
    "TextBlock",
    "Tool",
    "VideoBlock",
    "configure",
    "get_logger",
    "run",
    "tool",
]
