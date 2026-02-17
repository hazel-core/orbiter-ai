# orbiter-core

The `orbiter-core` package is the foundation of the Orbiter framework. It provides all the building blocks for creating and running LLM-powered agents: the type system, configuration, tool abstractions, the agent class, execution runners, multi-agent swarms, lifecycle hooks, events, and more.

## Installation

```bash
pip install orbiter-core
```

## Import Patterns

The top-level `orbiter` namespace re-exports the most commonly used classes:

```python
from orbiter import Agent, Tool, FunctionTool, tool, run, Swarm
from orbiter import ParallelGroup, SerialGroup, SwarmNode
```

For more specific imports, use the individual modules:

```python
from orbiter.types import UserMessage, AssistantMessage, RunResult, StreamEvent
from orbiter.config import ModelConfig, AgentConfig, RunConfig, parse_model_string
from orbiter.registry import Registry, RegistryError, agent_registry, tool_registry
from orbiter.events import EventBus, EventHandler
from orbiter.hooks import HookPoint, HookManager, Hook, run_hooks
from orbiter.tool import Tool, FunctionTool, tool, ToolError
from orbiter.agent import Agent, AgentError
from orbiter.runner import run
from orbiter.swarm import Swarm, SwarmError
from orbiter.human import HumanInputTool, HumanInputHandler, ConsoleHandler
from orbiter.loader import load_agents, load_swarm, register_agent_class, LoaderError
from orbiter.skills import Skill, SkillRegistry, ConflictStrategy, SkillError
```

## Namespace Package Architecture

`orbiter-core` uses `pkgutil.extend_path()` to create a namespace package. This allows other Orbiter packages (such as `orbiter-models`, `orbiter-context`) to add modules to the `orbiter` namespace without conflicts.

## Modules

| Module | Description |
|--------|-------------|
| [types](types.md) | Core message types, agent I/O, run results, streaming events |
| [config](config.md) | Configuration dataclasses for models, agents, tasks, and runs |
| [registry](registry.md) | Generic named registry with duplicate detection |
| [events](events.md) | Async event bus for decoupled communication |
| [hooks](hooks.md) | Lifecycle hook system for agent execution interception |
| [tool](tool.md) | Tool ABC, FunctionTool wrapper, @tool decorator |
| [agent](agent.md) | The Agent class -- the core autonomous unit |
| [runner](runner.md) | `run()`, `run.sync()`, `run.stream()` execution API |
| [swarm](swarm.md) | Multi-agent orchestration with workflow, handoff, and team modes |
| [human](human.md) | Human-in-the-loop tool and input handlers |
| [loader](loader.md) | YAML-based agent and swarm loading |
| [skills](skills.md) | Multi-source skill registry |

## Quick Example

```python
import asyncio
from orbiter import Agent, tool, run

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}."

agent = Agent(
    name="weather_bot",
    model="openai:gpt-4o",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)

result = run.sync(agent, "What is the weather in Tokyo?")
print(result.output)
```
