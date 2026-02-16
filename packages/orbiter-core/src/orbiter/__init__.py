"""Orbiter Core: Agent, Tool, Runner, Config, Events, Hooks, Swarm."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

from orbiter._internal.agent_group import ParallelGroup, SerialGroup
from orbiter._internal.nested import SwarmNode
from orbiter.agent import Agent
from orbiter.runner import run
from orbiter.swarm import Swarm
from orbiter.tool import FunctionTool, Tool, tool

__all__ = [
    "Agent",
    "FunctionTool",
    "ParallelGroup",
    "SerialGroup",
    "Swarm",
    "SwarmNode",
    "Tool",
    "run",
    "tool",
]
