# Changelog

All notable changes to Orbiter are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Live Message Injection** -- `Agent.inject_message(content)` pushes a `UserMessage` into a running agent's context, picked up before the next LLM call. Enables mid-run steering without cancelling. (`orbiter-core`)
- **`MessageInjectedEvent`** -- New streaming event type emitted by `run.stream()` when an injected message is drained. Always emitted (not gated by `detailed`). (`orbiter-core`)
- **`POST /inject` endpoint** -- HTTP endpoint on `orbiter-server` for injecting messages into a running agent via the `InjectRequest` model. (`orbiter-server`)

## [0.1.0] - 2025-02-16

Initial release of Orbiter, a ground-up rewrite of the AWorld multi-agent framework.

### Core Framework (`orbiter-core`)

- **Agent** -- Single `Agent` class replacing AWorld's 5 agent types (`LLMAgent`, `TaskLLMAgent`, `LoopLLMAgent`, `ParallelLLMAgent`, `SerialLLMAgent`). Supports tools, handoffs, lifecycle hooks, structured output, and configurable max steps.
- **Tool** -- `@tool` decorator for turning functions into LLM-callable tools with auto-generated JSON schemas from type hints and docstrings. `Tool` ABC for complex tools. Sync functions auto-wrapped via `asyncio.to_thread()`.
- **Swarm** -- Multi-agent orchestration with three modes: `"workflow"` (sequential pipeline via flow DSL), `"handoff"` (agent-driven delegation), and `"team"` (lead-worker pattern with auto-generated delegate tools).
- **Runner** -- Three entry points: `run()` (async), `run.sync()` (blocking), `run.stream()` (async generator). State tracking, loop detection, and retry logic built in.
- **Types** -- Typed message classes (`UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolResult`) as frozen Pydantic models. `RunResult`, `StreamEvent`, `Usage`, `AgentOutput`.
- **Config** -- Pydantic v2 models: `AgentConfig`, `ModelConfig`, `TaskConfig`, `RunConfig`. `parse_model_string()` for `"provider:model_name"` format.
- **Registry** -- Generic `Registry[T]` with fail-fast duplicate detection. Pre-built `agent_registry` and `tool_registry`.
- **Events** -- `EventBus` for decoupled async pub/sub communication.
- **Hooks** -- `HookManager` with `HookPoint` enum (`START`, `FINISHED`, `ERROR`, `PRE_LLM_CALL`, `POST_LLM_CALL`, `PRE_TOOL_CALL`, `POST_TOOL_CALL`).
- **Internal** -- `message_builder` (correct-by-construction message ordering), `output_parser` (LLM response to AgentOutput bridging, structured output validation), `call_runner` (execution loop with state tracking and loop detection), `state` (RunState/RunNode lifecycle), `graph` (flow DSL parser, topological sort).

### LLM Models (`orbiter-models`)

- **ModelProvider** -- Abstract base class with `complete()` and `stream()` methods.
- **OpenAIProvider** -- Full implementation for OpenAI's chat completions API.
- **AnthropicProvider** -- Full implementation for Anthropic's messages API.
- **Model Registry** -- `get_provider("openai:gpt-4o")` factory for building providers from model strings.
- **Types** -- `ModelResponse`, `StreamChunk`, `ToolCallDelta`, `FinishReason`, `ModelError`.

### Package Structure

- UV workspace monorepo with 13 packages under `packages/`
- Namespace packages via `pkgutil.extend_path()` for unified `orbiter.*` import path
- Zero heavy dependencies in `orbiter-core` (only `pydantic`)
- Provider SDKs isolated in `orbiter-models`

### Design Highlights

- Async-first with single sync bridge (`run.sync()`)
- Parallel tool execution via `asyncio.TaskGroup`
- Typed message union (`Message = UserMessage | AssistantMessage | SystemMessage | ToolResult`)
- Frozen Pydantic models for all data types
- Flow DSL for declarative multi-agent pipelines (`"a >> b >> c"`, `"a >> (b | c) >> d"`)
- Structured exception hierarchy rooted at `OrbiterError`
- Tool errors captured as `ToolResult(error=...)`, not propagated
- Exponential backoff retry for transient LLM errors

### Planned Packages (stubs created)

- `orbiter-context` -- Context engine: hierarchical state, prompt neurons, processors, workspace, RAG
- `orbiter-memory` -- Short-term and long-term memory backends
- `orbiter-mcp` -- Model Context Protocol client and server
- `orbiter-sandbox` -- Local and Kubernetes sandboxed execution
- `orbiter-trace` -- OpenTelemetry-based tracing
- `orbiter-eval` -- Evaluation framework with scorers and reflection
- `orbiter-a2a` -- Agent-to-Agent protocol
- `orbiter-cli` -- Command-line interface
- `orbiter-server` -- HTTP server for agent serving
- `orbiter-train` -- Training: trajectory collection and trainers

[0.1.0]: https://github.com/anthropics/orbiter/releases/tag/v0.1.0
