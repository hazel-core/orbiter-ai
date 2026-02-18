# orbiter-core Internal Modules

> **Stability Warning:** These are internal APIs, subject to change without notice. They are documented here for contributors and advanced users who need to understand the framework's internals. Do not depend on these APIs in production code -- use the public API from `orbiter.*` instead.

## Overview

The `orbiter._internal` package contains the implementation machinery that powers the public `orbiter` API. These modules handle message construction, response parsing, execution state tracking, the core run loop, handler abstractions, agent group primitives, nesting support, graph algorithms, and background task management.

## Import Pattern

```python
from orbiter._internal.message_builder import build_messages
from orbiter._internal.output_parser import parse_response
from orbiter._internal.state import RunState, RunNode, RunNodeStatus
from orbiter._internal.call_runner import call_runner
from orbiter._internal.handlers import Handler, AgentHandler, ToolHandler, GroupHandler
from orbiter._internal.agent_group import ParallelGroup, SerialGroup
from orbiter._internal.nested import SwarmNode
from orbiter._internal.graph import Graph, parse_flow_dsl, topological_sort
from orbiter._internal.background import BackgroundTaskHandler, BackgroundTask, PendingQueue
```

Note: `ParallelGroup`, `SerialGroup`, and `SwarmNode` are re-exported from `orbiter.__init__` as public API.

## Modules

| Module | Description |
|--------|-------------|
| [message_builder](message-builder.md) | Build ordered message lists for LLM provider calls |
| [output_parser](output-parser.md) | Parse LLM responses into agent-level output types |
| [state](state.md) | Run state tracking with execution nodes and lifecycle management |
| [call_runner](call-runner.md) | Core execution loop with state tracking and loop detection |
| [handlers](handlers.md) | Handler abstractions for composable agent, tool, and group execution |
| [agent_group](agent-group.md) | ParallelGroup and SerialGroup execution primitives |
| [nested](nested.md) | SwarmNode for nesting swarms within swarms |
| [graph](graph.md) | Directed graph, topological sort, and flow DSL parser |
| [background](background.md) | Background task handler with hot-merge and wake-up-merge patterns |
