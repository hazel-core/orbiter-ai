# Migration Guide

This section covers migrating from AWorld to Orbiter. Orbiter is a ground-up rewrite of AWorld that simplifies the API surface while preserving the framework's capabilities.

## Why Migrate?

AWorld grew to 96,500 lines of code with accumulated complexity:

- **5 agent types** that resist composition (`LLMAgent`, `TaskLLMAgent`, `LoopLLMAgent`, `ParallelLLMAgent`, `SerialLLMAgent`)
- **3 config systems** (`ConfigDict`, `BaseConfig`, Pydantic models) mixed throughout the codebase
- **Sync/async duplication** (`BaseTool` and `AsyncBaseTool`, `run()` and `sync_run()`)
- **Stringly-typed message routing** (`category="tool"`, `topic="GROUP_RESULTS"`)
- **Deep factory chains** (`Factory[T]` + `AgentManager` + `ToolsManager`)

Orbiter addresses all of these with:

- **1 Agent class** with composable behavior via Swarm modes
- **1 config system** (Pydantic v2 models)
- **Async-first** with automatic sync wrapping
- **Typed message classes** (`UserMessage`, `AssistantMessage`, `ToolResult`)
- **1 Registry** pattern (`Registry[T]`)

## Migration Scope

The migration is broken into these areas:

| Area | AWorld Package | Orbiter Package | Difficulty |
|------|---------------|-----------------|------------|
| Agent definition | `aworld.agents` | `orbiter.agent` | Low |
| Tool registration | `aworld.core.tool`, `aworld.tools` | `orbiter.tool` | Low |
| Running agents | `aworld.runner`, `aworld.runners` | `orbiter.runner` | Low |
| Configuration | `aworld.config.conf` | `orbiter.config` | Medium |
| Multi-agent | `aworld.agents.swarm_composer_agent` | `orbiter.swarm` | Medium |
| Context engine | `aworld.core.context.amni` | `orbiter.context` | High |
| Memory | `aworld.memory` | `orbiter.memory` | Medium |
| Models/LLM | `aworld.models` | `orbiter.models` | Low |
| Tracing | `aworld.trace` | `orbiter.trace` | Medium |
| Evaluation | `aworld.evaluations` | `orbiter.eval` | Medium |
| MCP | `aworld.mcp_client` | `orbiter.mcp` | Low |
| Sandbox | `aworld.sandbox` | `orbiter.sandbox` | Medium |

## Getting Started

1. Start with the [detailed migration guide](./from-aworld.md) for step-by-step instructions
2. Migrate the simplest agents first (single agent, no context, no memory)
3. Add tools, then multi-agent, then context/memory

## Package Installation

AWorld was a monolith. Orbiter is modular -- install only what you need:

```bash
# Minimal -- core + models
pip install orbiter-core orbiter-models

# Or the meta-package for everything
pip install orbiter

# With specific extras
pip install orbiter-memory[qdrant]
pip install orbiter-sandbox[kubernetes]
```
