# PRD: Orbiter Framework — Remaining Implementation (Phases 5–22)

## Introduction

Orbiter is a modern, minimal multi-agent framework being rewritten from AWorld (96,500 LOC). It is a UV workspace monorepo with 13 packages under `packages/`. Phases 0–4 are complete (288 tests passing), establishing the monorepo infrastructure, core types, config, registry, events, hooks, tool system, and LLM models package.

This PRD covers all remaining implementation work: Phases 5–22 (~106 sessions), taking Orbiter from foundational types to a fully functional agent framework with runner, swarm orchestration, context engine, memory, MCP, sandbox, tracing, evaluation, A2A protocol, CLI, server, training, examples, and CI.

Each user story maps 1:1 to a session in `docs/rewrite-plan.md` and is designed for Claude Code to implement in a single focused session (~100–150 lines of new code).

## Goals

- Complete the Orbiter framework with all 13 packages fully implemented and tested
- Maintain zero-broken-state between sessions — every session ends with all tests passing
- Achieve clean, minimal code: ~200 lines max per source file, async-first, Pydantic v2 only
- Rewrite clean from scratch — no copy-paste from `old/` reference code
- Deliver a framework that is simpler than AWorld while retaining all essential capabilities
- Provide comprehensive examples covering quick-start, multi-agent patterns, and benchmarks

## User Stories

---

### Phase 5: Agent Core

---

### US-001: Implement message builder

**Description:** As a framework developer, I want a message builder that constructs correctly ordered LLM message lists from system instructions, conversation history, and tool results, so that agents can communicate with LLM providers without message ordering bugs.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/_internal/__init__.py` (empty)
- [ ] Create `packages/orbiter-core/src/orbiter/_internal/message_builder.py`
- [ ] Implement `build_messages(instructions, history, tool_results)` function
- [ ] Handle message ordering: system → user → assistant → tool result cycles
- [ ] Handle edge cases: empty history, no tool results, multiple tool results
- [ ] ~120 lines of source code
- [ ] Create/update `packages/orbiter-core/tests/test_message_builder.py`
- [ ] Tests cover: empty history, single turn, multi-turn, tool result insertion, message ordering validation
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/agents/llm_agent.py` lines 362–500 (`async_messages_transform`)

---

### US-002: Implement output parser

**Description:** As a framework developer, I want an output parser that extracts text, tool calls, and structured output from LLM responses, so that the agent can decide its next action.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/_internal/output_parser.py`
- [ ] Implement `parse_response(response: ModelResponse) -> ParsedOutput`
- [ ] `ParsedOutput` contains: `text`, `tool_calls`, `structured_output`, `finish_reason`
- [ ] Handle text-only responses, tool-call-only responses, and mixed responses
- [ ] Handle structured output via Pydantic model validation when `output_type` is provided
- [ ] ~100 lines of source code
- [ ] Create/update `packages/orbiter-core/tests/test_output_parser.py`
- [ ] Tests cover: text-only, tool-call-only, mixed, structured output validation, malformed responses
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/core/model_output_parser/` (~150 LOC)

---

### US-003: Implement Agent class — init and configuration

**Description:** As a framework user, I want to create Agent instances with a clean constructor that accepts name, model, instructions, tools, hooks, memory, handoffs, and output_type, so that I can define agents declaratively.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/agent.py`
- [ ] `Agent.__init__` accepts: `name` (required), `model`, `instructions`, `tools`, `handoffs`, `hooks`, `output_type`, `max_steps`, `temperature`, `max_tokens`, `memory`, `context`
- [ ] All parameters are keyword-only except `name`
- [ ] Sensible defaults per design spec (model=`"openai:gpt-4o"`, max_steps=10, temperature=1.0)
- [ ] Tool registration validates tool types
- [ ] `describe()` method returns agent metadata (name, model, tools, capabilities)
- [ ] ~100 lines of source code
- [ ] Create `packages/orbiter-core/tests/test_agent.py`
- [ ] Tests cover: minimal agent, full-featured agent, config validation, describe() output
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/agents/llm_agent.py` (1,156 LOC), `old/aworld/core/agent/base.py` (451 LOC)

---

### US-004: Implement Agent run method with retry logic

**Description:** As a framework developer, I want the Agent to have a `run()` method that builds messages, calls the LLM, parses the response, and returns the result — with retry logic for transient errors, so that single-turn agent execution is reliable.

**Acceptance Criteria:**
- [ ] Add `async run(input, messages, provider)` method to `Agent`
- [ ] Wires message_builder → LLM call → output_parser → return
- [ ] Single-turn execution only (no tool loop yet)
- [ ] Hooks: `PRE_LLM_CALL`, `POST_LLM_CALL` fired at correct points
- [ ] Retry logic: configurable `max_retries` (default 3) with exponential backoff
- [ ] Context-length errors (`ModelError` with specific code) fail immediately, no retry
- [ ] Transient errors (rate limit, timeout, server error) retry with backoff
- [ ] ~130 lines of source code
- [ ] Tests cover: successful run, retry on transient error, immediate fail on context-length error, hook invocation order
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/agents/llm_agent.py` lines 868–983 (`invoke_model`)

---

### US-005: Implement Agent tool execution loop

**Description:** As a framework developer, I want the Agent to execute tool calls returned by the LLM, feed results back, and re-call the LLM in a loop until a text response is produced or max_steps is reached, so that agents can use tools to accomplish tasks.

**Acceptance Criteria:**
- [ ] Extend `Agent.run()`: if LLM returns tool calls → execute tools → feed results back → re-call LLM
- [ ] `max_steps` guard prevents infinite loops
- [ ] `PRE_TOOL_CALL` and `POST_TOOL_CALL` hooks fired for each tool execution
- [ ] Parallel tool execution via `asyncio.TaskGroup` when multiple tool calls are returned
- [ ] Tool errors caught and returned as `ToolResult(error=...)`, not propagated
- [ ] ~120 lines of source code
- [ ] Tests cover: single tool call, multi-tool parallel call, tool error handling, max_steps enforcement
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/agents/llm_agent.py` lines 623–739 (`async_policy`)

---

### US-006: Agent edge case tests

**Description:** As a framework developer, I want comprehensive tests for Agent edge cases so that the implementation is robust.

**Acceptance Criteria:**
- [ ] Update `packages/orbiter-core/tests/test_agent.py`
- [ ] Tests cover: multi-tool calls in sequence, parallel tool calls, tool returning error, max_steps reached, retry behavior across tool loop, agent with no tools, agent with handoffs declared (preparation for Phase 7)
- [ ] ~120 lines of test code
- [ ] `uv run pytest` — all tests pass
- [ ] Total agent tests: ~15-20

---

### US-007: Implement Human-in-the-loop tool

**Description:** As a framework user, I want a HITL tool that pauses agent execution to request user confirmation or input, so that I can build agents with human oversight.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/human.py`
- [ ] `HumanInputTool` — async tool that blocks for user input
- [ ] `HumanInputHandler` protocol with `async get_input(prompt: str) -> str`
- [ ] `ConsoleHandler` — default interactive console handler (reads from stdin)
- [ ] Tool schema generated correctly (prompt parameter, optional choices)
- [ ] ~120 lines of source code
- [ ] Create `packages/orbiter-core/tests/test_human.py`
- [ ] Tests cover: tool schema, handler invocation with mocked input, timeout behavior
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/tools/human/human.py` (115 LOC)

---

### Phase 6: Runner & Execution

---

### US-008: Implement run state tracking

**Description:** As a framework developer, I want internal state tracking for run execution (messages, iterations, status) so that the runner can manage agent lifecycle.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/_internal/state.py`
- [ ] `RunState` — tracks messages, tool calls, iterations, current agent, status
- [ ] `RunNodeStatus` enum: INIT, RUNNING, SUCCESS, FAILED, TIMEOUT
- [ ] `RunNode` — per-step state with agent_id, group tracking, timestamps
- [ ] ~120 lines of source code
- [ ] Create `packages/orbiter-core/tests/test_state.py`
- [ ] Tests cover: state transitions, node lifecycle, message accumulation
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/runners/state_manager.py` (841 LOC)

---

### US-009: Implement call runner core loop

**Description:** As a framework developer, I want the internal call runner that orchestrates the LLM→tool→LLM loop with state tracking and loop detection, so that the public `run()` function has a reliable execution engine.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/_internal/call_runner.py`
- [ ] `async call_runner(agent, input, state)` — the core execution loop
- [ ] Integrates with `Agent.run()`, `RunState` tracking
- [ ] Endless loop detection with configurable threshold
- [ ] ~120 lines of source code
- [ ] Create `packages/orbiter-core/tests/test_call_runner.py`
- [ ] Tests cover: single-turn, multi-turn, loop detection, state updates
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/runners/call_driven_runner.py` (851 LOC)

---

### US-010: Implement public run() entry point

**Description:** As a framework user, I want `run(agent, input)` and `run.sync(agent, input)` as the primary API for executing agents, so that running agents is simple and intuitive.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/runner.py`
- [ ] `async run(agent_or_swarm, input, messages=None)` → `RunResult`
- [ ] `run.sync(agent_or_swarm, input)` — sync wrapper using `asyncio.run()`
- [ ] Wraps bare `Agent` in single-agent `Swarm` internally if needed
- [ ] Returns `RunResult` with `output`, `messages`, `usage`, `steps`
- [ ] ~100 lines of source code
- [ ] Create `packages/orbiter-core/tests/test_runner.py`
- [ ] Tests cover: `run()` async, `run.sync()`, multi-turn via messages param, error propagation
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/runner.py` (320 LOC)

---

### US-011: Implement streaming run

**Description:** As a framework user, I want `run.stream(agent, input)` that yields `StreamEvent` objects in real-time, so that I can build streaming UIs and show progressive agent output.

**Acceptance Criteria:**
- [ ] Add `run.stream()` as an async generator yielding `StreamEvent`
- [ ] Yields `TextEvent` for text chunks, `ToolCallEvent` for tool invocations
- [ ] Integrates with provider streaming (`provider.stream()`)
- [ ] ~120 lines of source code
- [ ] Tests cover: streaming text events in order, tool call events, stream completion
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

---

### US-012: Implement handler system — base + agent handler

**Description:** As a framework developer, I want a handler abstraction and an AgentHandler that routes between agents in a swarm, so that multi-agent execution is composable.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/_internal/handlers.py`
- [ ] `Handler[IN, OUT]` ABC with `async handle()` → `AsyncGenerator`
- [ ] `AgentHandler` — routes between agents, handles handoff dispatch
- [ ] Swarm topology-aware stop checks (workflow/handoff/team modes)
- [ ] ~130 lines of source code
- [ ] Tests cover: handler dispatch, agent routing, handoff detection
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/runners/handler/` (~2,400 LOC, 10 handler types)

---

### US-013: Implement tool + group handlers

**Description:** As a framework developer, I want ToolHandler and GroupHandler for dynamic tool loading/execution and parallel agent group execution, so that the runner can handle complex execution patterns.

**Acceptance Criteria:**
- [ ] Add `ToolHandler` to `handlers.py` — dynamic tool loading, execution, result aggregation
- [ ] Add `GroupHandler` — parallel agent/tool group execution with dependency resolution
- [ ] ~130 lines of source code
- [ ] Tests cover: parallel tool execution, group execution ordering, dependency resolution
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

---

### US-014: Implement background task handler

**Description:** As a framework developer, I want a background task handler that supports hot-merge (running task) and wake-up-merge (checkpoint restore) patterns, so that long-running tasks can integrate background results.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/_internal/background.py`
- [ ] `BackgroundTaskHandler` with hot-merge and wake-up-merge patterns
- [ ] Pending message queue for background results
- [ ] Integration point with checkpoint system
- [ ] ~120 lines of source code
- [ ] Tests cover: hot-merge, wake-up-merge, pending message handling
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/runners/handler/background_task.py`

---

### US-015: Runner integration tests

**Description:** As a framework developer, I want end-to-end integration tests for the runner system so that Agent + Tool + run() works correctly together.

**Acceptance Criteria:**
- [ ] End-to-end tests: `Agent` + `@tool` + `run()` with mocked LLM provider
- [ ] Handler pipeline tests
- [ ] Background task scenario tests
- [ ] ~100 lines of test code
- [ ] `uv run pytest` — all tests pass for orbiter-core
- [ ] Update `packages/orbiter-core/src/orbiter/__init__.py` with exports: `Agent`, `run`, `tool`, `Tool`

---

### Phase 7: Swarm / Multi-Agent

---

### US-016: Implement graph utilities

**Description:** As a framework developer, I want graph utilities (topological sort, cycle detection, flow DSL parsing) so that swarm orchestration can define agent execution order.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/_internal/graph.py`
- [ ] Simple adjacency list graph implementation
- [ ] `topological_sort()` using Kahn's algorithm
- [ ] Cycle detection (raises on cyclic graphs)
- [ ] `parse_flow_dsl("a >> b >> c")` → list of edges
- [ ] ~100 lines of source code
- [ ] Create `packages/orbiter-core/tests/test_graph.py`
- [ ] Tests cover: topo sort, cycle detection, DSL parsing, complex DAGs
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

---

### US-017: Implement Swarm — workflow mode

**Description:** As a framework user, I want `Swarm(agents=[...], flow="a >> b >> c")` that runs agents sequentially, passing output as input, so that I can build agent pipelines.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/swarm.py`
- [ ] `Swarm.__init__` accepts: `agents`, `flow` (DSL string), `mode` (default "workflow")
- [ ] `mode="workflow"`: execute agents in topological order from flow DSL
- [ ] Output of each agent becomes input for the next
- [ ] ~120 lines of source code
- [ ] Create `packages/orbiter-core/tests/test_swarm.py`
- [ ] Tests cover: sequential execution, output→input chaining, 3-agent pipeline
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/core/agent/swarm.py` (1,211 LOC)

---

### US-018: Implement Swarm — handoff mode

**Description:** As a framework user, I want agents to be able to hand off to other agents dynamically (like triage → billing), so that I can build agent-driven delegation patterns.

**Acceptance Criteria:**
- [ ] Add `mode="handoff"` support to `Swarm`
- [ ] Agents can return a handoff action targeting another agent in the swarm
- [ ] Handoff transfers control and conversation history
- [ ] Endless loop detection with configurable threshold
- [ ] ~120 lines of source code
- [ ] Tests cover: agent A hands off to agent B, loop detection triggers, handoff with conversation history
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

---

### US-019: Implement Swarm — team mode

**Description:** As a framework user, I want a team mode where a lead agent coordinates worker agents, so that I can build leader-delegation patterns.

**Acceptance Criteria:**
- [ ] Add `mode="team"` support to `Swarm`
- [ ] First agent in list is the lead; others are workers
- [ ] Lead can delegate to workers and synthesize their results
- [ ] ~120 lines of source code
- [ ] Tests cover: lead delegates to worker, worker responds, lead synthesizes final output
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

---

### US-020: Implement parallel + serial agent groups

**Description:** As a framework developer, I want parallel and serial agent group primitives so that swarm flow DSL can express `"(a | b) >> c"` for concurrent-then-sequential execution.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/_internal/agent_group.py`
- [ ] `ParallelGroup` — concurrent agent execution via `asyncio.TaskGroup`, custom result aggregation
- [ ] `SerialGroup` — dependency-based sequential execution with output→input chaining
- [ ] Both integrate as nodes in Swarm flow DSL
- [ ] ~120 lines of source code
- [ ] Tests cover: parallel execution, serial chaining, mixed topologies
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/agents/parallel_llm_agent.py`, `old/aworld/agents/serial_llm_agent.py`

---

### US-021: Implement nested swarms

**Description:** As a framework user, I want to use a Swarm as an agent within another Swarm, enabling hierarchical multi-agent composition.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/_internal/nested.py`
- [ ] `Swarm` can be used as a node in another `Swarm`'s agent list
- [ ] Recursive execution with context isolation
- [ ] ~100 lines of source code
- [ ] Tests cover: 2-level nested swarm, context isolation between levels
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/agents/task_llm_agent.py` (62 LOC)

---

### US-022: Swarm integration + public API wiring

**Description:** As a framework user, I want `run(swarm, input)` to work seamlessly and `Swarm` exported from `orbiter`, so that multi-agent orchestration is accessible from the top-level API.

**Acceptance Criteria:**
- [ ] Wire `Swarm` into `run()` — detect Agent vs Swarm input
- [ ] Update `packages/orbiter-core/src/orbiter/__init__.py` — export `Swarm`
- [ ] Integration tests for all swarm modes: workflow, handoff, team
- [ ] Integration tests for parallel/serial groups and nesting
- [ ] ~100 lines of source code + tests
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/orbiter-core/` — zero type errors
- [ ] `uv run pytest` — all tests pass

---

### Phase 8: Context Engine

---

### US-023: Implement ContextConfig + ContextState

**Description:** As a framework developer, I want ContextConfig (automation settings) and ContextState (hierarchical key-value state with parent inheritance) as the foundation of the context engine.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-context/src/orbiter/context/config.py`
- [ ] `ContextConfig` — Pydantic v2 frozen model: `mode` ("pilot"|"copilot"|"navigator"), `history_rounds`, `summary_threshold`, `offload_threshold`, `enable_retrieval`, `neuron_names`
- [ ] Create `packages/orbiter-context/src/orbiter/context/state.py`
- [ ] `ContextState` — `get(key, default)` searches local then parent chain, `set(key, value)` writes local only, `local_dict()`, `to_dict()`
- [ ] ~120 lines of source code total
- [ ] Create `packages/orbiter-context/tests/test_context_config.py` and `test_context_state.py`
- [ ] Tests cover: config creation/validation/defaults, state inheritance, local vs parent, merge
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/core/context/context_state.py` (232 LOC), `old/aworld/core/context/amni/config.py` (~200 LOC)

---

### US-024: Implement Context class — core lifecycle

**Description:** As a framework developer, I want the Context class with fork/merge for hierarchical task decomposition, so that multi-agent contexts are isolated but composable.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-context/src/orbiter/context/context.py`
- [ ] `Context.__init__`: `task_id` (required), `config`, `parent`
- [ ] `fork(task_id)` — create child context with state inheritance
- [ ] `merge(child)` — consolidate child state back into parent with net token calculation
- [ ] ~130 lines of source code
- [ ] Create `packages/orbiter-context/tests/test_context.py`
- [ ] Tests cover: context creation, fork, merge, hierarchical state isolation
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/core/context/base.py` (913 LOC)

---

### US-025: Implement TokenTracker

**Description:** As a framework developer, I want per-agent, per-step token tracking so that cost analysis and budget enforcement are possible.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-context/src/orbiter/context/token_tracker.py`
- [ ] `TokenTracker` with `add_step(agent_id, prompt_tokens, output_tokens)`
- [ ] `TokenStep` dataclass: prompt_tokens, output_tokens, step index
- [ ] `get_trajectory(agent_id)` → list of steps
- [ ] `total_usage()` → aggregated prompt/output/total tokens
- [ ] ~100 lines of source code
- [ ] Create `packages/orbiter-context/tests/test_token_tracker.py`
- [ ] Tests cover: step tracking, trajectory retrieval, multi-agent aggregation
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

---

### US-026: Implement Neuron base + core built-in neurons

**Description:** As a framework developer, I want a Neuron ABC and core built-in neurons (system, task, history) so that prompts can be composed from modular components.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-context/src/orbiter/context/neuron.py`
- [ ] `Neuron` ABC with `async format(ctx) -> str` and `priority: int`
- [ ] `neuron_registry` — `Registry[Neuron]` for neuron discovery
- [ ] `SystemNeuron` (priority 100) — date/time/platform dynamic variables
- [ ] `TaskNeuron` (priority 1) — task ID, input, output, subtask plan
- [ ] `HistoryNeuron` (priority 10) — conversation history with windowing
- [ ] ~130 lines of source code
- [ ] Create `packages/orbiter-context/tests/test_neuron.py`
- [ ] Tests cover: neuron formatting, priority ordering, registry, each built-in neuron output
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/core/context/amni/prompt/neurons/` (~600 LOC)

---

### US-027: Implement extended neurons

**Description:** As a framework developer, I want additional neurons (skills, workspace, todo, facts, entities, knowledge) and a dynamic variable system, so that the prompt builder has rich context sources.

**Acceptance Criteria:**
- [ ] Add to `neuron.py` or create `packages/orbiter-context/src/orbiter/context/_internal/neurons.py`
- [ ] `SkillNeuron` (priority 40), `WorkspaceNeuron` (priority 30), `TodoNeuron` (priority 2), `FactNeuron` (priority 50), `EntityNeuron` (priority 60), `KnowledgeNeuron` (priority 20)
- [ ] Dynamic variable system: `DynamicVariableRegistry` with nested path resolution
- [ ] ~130 lines of source code
- [ ] Tests cover: each neuron type produces correct prompt fragment
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

---

### US-028: Implement PromptBuilder

**Description:** As a framework user, I want a PromptBuilder that composes neurons in priority order to build rich system prompts, so that agent prompts are constructed from modular, reusable components.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-context/src/orbiter/context/prompt_builder.py`
- [ ] `PromptBuilder.__init__(ctx: Context)`
- [ ] `add(neuron_name, **kwargs)` — register a neuron for building
- [ ] `async build() -> str` — resolve all neurons in priority order, compose final prompt
- [ ] Template variable resolution with hierarchical context traversal
- [ ] ~120 lines of source code
- [ ] Create `packages/orbiter-context/tests/test_prompt_builder.py`
- [ ] Tests cover: add neurons, build prompt, priority ordering, variable resolution
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

---

### US-029: Implement ContextProcessor pipeline

**Description:** As a framework developer, I want event-driven context processors that intervene at specific points in the LLM execution cycle (pre_llm_call, post_tool_call, etc.), so that context can be dynamically transformed.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-context/src/orbiter/context/processor.py`
- [ ] `ContextProcessor` ABC with `event: str` and `async process(ctx, payload) -> None`
- [ ] `ProcessorPipeline` — registers processors, fires them by event type
- [ ] Built-in: `SummarizeProcessor` (pre_llm_call), `ToolResultOffloader` (post_tool_call)
- [ ] ~130 lines of source code
- [ ] Create `packages/orbiter-context/tests/test_context_processor.py`
- [ ] Tests cover: processor registration, event filtering, pipeline execution order, built-in processors
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/core/context/amni/processor/` (~400 LOC)

---

### US-030: Implement Workspace + artifact system

**Description:** As a framework user, I want a Workspace for persistent artifact storage during execution, with versioning and observer notifications, so that agents can read/write files and large tool results can be offloaded.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-context/src/orbiter/context/workspace.py`
- [ ] `Workspace` — `write(name, content)`, `read(name)`, `list()`, `delete(name)`
- [ ] `ArtifactType` enum: TEXT, CODE, MARKDOWN, JSON, CSV, IMAGE
- [ ] Artifact versioning: `version_history(name)`, `revert_to_version(name, version)`
- [ ] Observer pattern: `on_create`, `on_update`, `on_delete` callbacks
- [ ] Local filesystem backend
- [ ] ~130 lines of source code
- [ ] Create `packages/orbiter-context/tests/test_workspace.py`
- [ ] Tests cover: write/read/list/delete, versioning, observer notifications
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/core/context/amni/worksapces.py` (~200 LOC)

---

### US-031: Implement workspace-retriever integration

**Description:** As a framework developer, I want artifacts added to workspace to be auto-indexed in the knowledge store for RAG retrieval, so that agents can search their own artifacts semantically.

**Acceptance Criteria:**
- [ ] Wire workspace into RAG pipeline: workspace writes auto-index in KnowledgeStore
- [ ] Chunk range queries for large artifacts
- [ ] ~100 lines of source code
- [ ] Tests cover: artifact → chunk → search round-trip
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

---

### US-032: Implement Checkpoint

**Description:** As a framework user, I want to save and restore complete execution state for long-running tasks, so that work can be resumed after interruption.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-context/src/orbiter/context/checkpoint.py`
- [ ] `Checkpoint` — serialized context snapshot (values, metadata, version)
- [ ] `Context.snapshot()` → `Checkpoint`
- [ ] `Context.restore(checkpoint)` → `Context`
- [ ] Checkpoints are versioned per session
- [ ] ~100 lines of source code
- [ ] Create `packages/orbiter-context/tests/test_checkpoint.py`
- [ ] Tests cover: snapshot, restore, version incrementing, state preservation
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

---

### US-033: Implement Knowledge store + RAG basics

**Description:** As a framework developer, I want a KnowledgeStore with artifact indexing and semantic search, so that the context engine can retrieve relevant knowledge.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-context/src/orbiter/context/_internal/__init__.py`
- [ ] Create `packages/orbiter-context/src/orbiter/context/_internal/knowledge.py`
- [ ] `KnowledgeStore` — `add(name, content)`, `search(query, top_k)`, `get(name)`, `get_range(name, start, end)`
- [ ] Basic chunking + in-memory vector store for testing
- [ ] ~130 lines of source code
- [ ] Tests cover: add/search/get artifacts, range queries, chunking
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/core/context/amni/retrieval/` (~800 LOC)

---

### US-034: Implement context tools

**Description:** As a framework user, I want context tools (planning, knowledge, file) that let agents manipulate their own context during execution, so that agents can plan, search knowledge, and manage files.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-context/src/orbiter/context/tools.py`
- [ ] `planning_tool` — `add_todo(item)`, `get_todo()` for task planning checklist
- [ ] `knowledge_tool` — `get_knowledge(name)`, `grep_knowledge(name, pattern)` for artifact search
- [ ] `file_tool` — `read_file(path)` from working directory
- [ ] All implemented as `@tool`-decorated functions
- [ ] ~120 lines of source code
- [ ] Create `packages/orbiter-context/tests/test_context_tools.py`
- [ ] Tests cover: each tool's execution, context state mutation
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/core/context/amni/tool/` (~400 LOC)

---

### US-035: Context package __init__ + integration tests

**Description:** As a framework user, I want the context package to have a clean public API and pass integration tests end-to-end.

**Acceptance Criteria:**
- [ ] Update `packages/orbiter-context/src/orbiter/context/__init__.py` — exports: `Context`, `ContextConfig`, `ContextState`, `PromptBuilder`, `ContextProcessor`, `Neuron`, `Workspace`
- [ ] Integration tests: Context + PromptBuilder + Processor + Workspace end-to-end
- [ ] Wire context into Agent (`agent.py` context parameter)
- [ ] ~100 lines of source code + tests
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pytest` — all tests pass for orbiter-context

---

### Phase 9: Memory Package

---

### US-036: Implement memory interface + types

**Description:** As a framework developer, I want a MemoryStore protocol and typed MemoryItem hierarchy so that memory backends are pluggable and type-safe.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-memory/src/orbiter/memory/base.py`
- [ ] `MemoryStore` protocol: `add()`, `get()`, `search()`, `clear()`
- [ ] `MemoryItem` with subtypes: `SystemMemory`, `HumanMemory`, `AIMemory`, `ToolMemory`
- [ ] `MemoryMetadata` with user_id, session_id, task_id, agent_id scoping
- [ ] Status lifecycle: DRAFT → ACCEPTED → DISCARD
- [ ] ~120 lines of source code
- [ ] Tests cover: types, status transitions, protocol conformance
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/memory/models.py` (592 LOC)

---

### US-037: Implement short-term memory

**Description:** As a framework developer, I want short-term memory that manages conversation context with scope-based filtering and truncation.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-memory/src/orbiter/memory/short_term.py`
- [ ] `ShortTermMemory` implementing `MemoryStore`
- [ ] Scope-based filtering: user, session, task
- [ ] Incomplete message pair filtering (tool call/response integrity)
- [ ] ~130 lines of source code
- [ ] Tests cover: add messages, truncation, windowing, scope filtering
- [ ] `uv run pytest` — all tests pass

---

### US-038: Implement summary + compression

**Description:** As a framework developer, I want summary trigger logic and multi-template summary generation so that long conversations can be compressed while preserving key information.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-memory/src/orbiter/memory/summary.py`
- [ ] Summary trigger: message count threshold, token count threshold
- [ ] Multi-template summaries: conversation, facts, profiles
- [ ] `SummaryConfig` — prompts, thresholds, compression rules
- [ ] ~130 lines of source code
- [ ] Tests cover: trigger detection, summary generation with mocked LLM
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/memory/main.py` (928 LOC — summary logic)

---

### US-039: Implement long-term memory orchestrator

**Description:** As a framework developer, I want long-term memory with async LLM extraction of user profiles, agent experiences, and facts, so that agents build persistent knowledge across sessions.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-memory/src/orbiter/memory/long_term.py`
- [ ] `LongTermMemory` — persistent memory across sessions
- [ ] `MemoryOrchestrator` — async extraction of UserProfile, AgentExperience, Facts
- [ ] Processing task queue with status tracking
- [ ] ~130 lines of source code
- [ ] Tests cover: extraction with mocked LLM, task queue lifecycle
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/memory/longterm/` (~260 LOC)

---

### US-040: Implement SQLite memory backend

**Description:** As a framework developer, I want a SQLite-backed memory store so that memory persists to disk without external dependencies.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-memory/src/orbiter/memory/backends/sqlite.py`
- [ ] `SQLiteMemoryStore` implementing `MemoryStore`
- [ ] JSON indexes for metadata fields, soft deletes, version field
- [ ] ~120 lines of source code
- [ ] Tests with temp SQLite database
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/memory/db/sqlite.py` (426 LOC)

---

### US-041: Implement Postgres memory backend

**Description:** As a framework developer, I want a Postgres-backed memory store for production deployments.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-memory/src/orbiter/memory/backends/postgres.py`
- [ ] `PostgresMemoryStore` implementing `MemoryStore`
- [ ] ~120 lines of source code
- [ ] Tests (may need mock or skip if no postgres available)
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/memory/db/postgres.py` (375 LOC)

---

### US-042: Implement embeddings + vector search

**Description:** As a framework developer, I want embedding-based vector search for semantic memory retrieval.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-memory/src/orbiter/memory/backends/vector.py`
- [ ] `VectorMemoryStore` wrapping embedding + vector DB
- [ ] `Embeddings` ABC with sync + async variants
- [ ] OpenAI-compatible embedding provider with dimension support
- [ ] ~130 lines of source code
- [ ] Tests with mocked embeddings
- [ ] `uv run pytest` — all tests pass

---

### US-043: Memory package __init__ + integration

**Description:** As a framework user, I want the memory package to have a clean public API and integrate with Agent.

**Acceptance Criteria:**
- [ ] Update `packages/orbiter-memory/src/orbiter/memory/__init__.py` — exports
- [ ] Wire memory into Agent
- [ ] Memory event integration (emit memory events for async processing)
- [ ] ~80 lines of source code
- [ ] `uv run pytest` — all tests pass for orbiter-memory

---

### Phase 10: MCP Integration

---

### US-044: MCP client — server connection

**Description:** As a framework developer, I want an MCP client that connects to MCP servers via SSE, stdio, and streamable-http transports.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-mcp/src/orbiter/mcp/client.py`
- [ ] Multiple transport types: SSE, stdio, streamable-http
- [ ] Server instance caching/reuse with session isolation
- [ ] ~130 lines of source code
- [ ] Tests cover: connection lifecycle, transport selection
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/mcp_client/utils.py` (1,460 LOC)

---

### US-045: MCP tools — loading + conversion

**Description:** As a framework developer, I want MCP tool schema extraction and conversion to Orbiter Tool format.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-mcp/src/orbiter/mcp/tools.py`
- [ ] Tool schema extraction from MCP server
- [ ] `mcp__` namespace mapping for tool names
- [ ] Tool black/white-list filtering
- [ ] ~120 lines of source code
- [ ] Tests cover: schema conversion, namespace mapping, filtering
- [ ] `uv run pytest` — all tests pass

---

### US-046: MCP server decorator

**Description:** As a framework user, I want an `@mcp_server()` class decorator that converts Python methods to MCP tools, so that I can expose tools as MCP servers.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-mcp/src/orbiter/mcp/server.py`
- [ ] `@mcp_server()` class decorator converting methods to MCP tools
- [ ] `MCPServerRegistry` for singleton server instances
- [ ] ~120 lines of source code
- [ ] Tests cover: decorator, method→tool conversion, registry
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/mcp_client/decorator.py` (233 LOC)

---

### US-047: MCP execution + tests

**Description:** As a framework developer, I want MCP tool execution with retry logic and integration tests.

**Acceptance Criteria:**
- [ ] Retry logic with configurable timeout
- [ ] Environment variable substitution in mcp.json config
- [ ] Integration tests for full MCP lifecycle
- [ ] ~100 lines of source code
- [ ] `uv run pytest` — all tests pass for orbiter-mcp

---

### Phase 10.5: Config-Driven Loading & Skill Registry

---

### US-048: YAML agent loader

**Description:** As a framework user, I want to define agents and swarms in YAML files with variable substitution, so that I can create agents without writing Python code.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/loader.py`
- [ ] Load agent/swarm definitions from YAML
- [ ] `${ENV_VAR}` and `${vars.KEY}` substitution
- [ ] Swarm topology patterns: workflow, handoff, team
- [ ] Agent factory dispatch (builtin vs. custom classes)
- [ ] ~130 lines of source code
- [ ] Tests cover: YAML parsing, variable substitution, swarm creation from YAML
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/config/agent_loader.py` (196 LOC)

---

### US-049: Skill registry

**Description:** As a framework user, I want a multi-source skill registry that loads skills from local paths and GitHub URLs, so that agents can discover and use reusable skill libraries.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-core/src/orbiter/skills.py`
- [ ] `SkillRegistry` — multi-source skill management
- [ ] GitHub URL parsing & shallow clone with branch support, cached at `~/.orbiter/skills/`
- [ ] YAML front-matter extraction (name, desc, tool_list, type, active)
- [ ] Conflict resolution strategies (keep_first, keep_last, raise)
- [ ] Search + filtering capabilities
- [ ] ~130 lines of source code
- [ ] Tests cover: local skill loading, registry operations, search, conflict resolution
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/utils/skill_loader.py` (822 LOC)

---

### Phase 11: Sandbox Package

---

### US-050: Sandbox interface + local sandbox

**Description:** As a framework developer, I want a Sandbox ABC and local implementation so that agents can execute code in controlled environments.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-sandbox/src/orbiter/sandbox/base.py`
- [ ] `SandboxStatus` enum: INIT, RUNNING, IDLE, ERROR, CLOSED
- [ ] `Sandbox` ABC with workspace, MCP integration, agent configuration
- [ ] `LocalSandbox` implementation
- [ ] ~130 lines of source code
- [ ] Tests cover: lifecycle, status transitions
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/sandbox/base.py` (628 LOC)

---

### US-051: Built-in sandbox tools

**Description:** As a framework user, I want sandboxed filesystem and terminal tools so that agents can interact with the local environment safely.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-sandbox/src/orbiter/sandbox/tools.py`
- [ ] `FilesystemTool` with `allowed_directories` sandboxing, safe path validation
- [ ] `TerminalTool` with dangerous command blacklist, platform detection, timeout
- [ ] ~130 lines of source code
- [ ] Tests cover: path validation, command filtering, sandbox boundaries
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/sandbox/builtin/filesystem.py`, `old/aworld/sandbox/builtin/terminal.py`

---

### US-052: Sandbox builder

**Description:** As a framework user, I want a fluent builder API for constructing sandboxes with method chaining.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-sandbox/src/orbiter/sandbox/builder.py`
- [ ] Fluent API with method chaining for sandbox construction
- [ ] Lazy evaluation: auto-build on first API call
- [ ] ~120 lines of source code
- [ ] Tests cover: builder chaining, lazy evaluation
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/sandbox/builder/sandbox_builder.py` (324 LOC)

---

### US-053: Kubernetes sandbox

**Description:** As a framework developer, I want a Kubernetes-based sandbox for remote execution environments.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-sandbox/src/orbiter/sandbox/kubernetes.py`
- [ ] `KubernetesSandbox` implementing `Sandbox`
- [ ] ~120 lines of source code
- [ ] Tests (may need mock)
- [ ] `uv run pytest` — all tests pass

---

### Phase 12: Trace Package

---

### US-054: Trace config + base

**Description:** As a framework developer, I want trace configuration and semantic conventions for agent/tool observability attributes.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-trace/src/orbiter/trace/config.py`
- [ ] `TraceConfig` with backend selection, sampling, export settings
- [ ] Semantic conventions for `gen_ai.*`, `agent.*`, `tool.*` attributes
- [ ] ~120 lines of source code
- [ ] Tests cover: config creation, convention constants
- [ ] `uv run pytest` — all tests pass

---

### US-055: Span decorator + context manager

**Description:** As a framework user, I want a `@traced` decorator that instruments sync/async functions with span creation and metadata extraction.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-trace/src/orbiter/trace/decorator.py`
- [ ] `@traced` decorator supporting sync, async, generators, async generators
- [ ] Function metadata extraction (qualname, module, line number, parameters)
- [ ] Stack frame analysis with user-code filtering
- [ ] ~120 lines of source code
- [ ] Tests cover: sync/async decoration, metadata extraction
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/trace/function_trace.py` (181 LOC)

---

### US-056: Agent/tool instrumentation

**Description:** As a framework developer, I want automatic agent and tool metrics (duration, counter, token usage) so that execution is observable.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-trace/src/orbiter/trace/instrumentation.py`
- [ ] Agent metrics: `agent_run_duration` (histogram), `agent_run_counter`, `agent_token_usage`
- [ ] Tool metrics: `tool_step_duration`, `tool_step_counter`
- [ ] ~120 lines of source code
- [ ] Tests cover: metric recording, histogram buckets
- [ ] `uv run pytest` — all tests pass

---

### US-057: Trace context propagation

**Description:** As a framework developer, I want W3C Baggage standard propagation for cross-service trace correlation.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-trace/src/orbiter/trace/propagation.py`
- [ ] W3C Baggage standard (RFC 9110) propagation
- [ ] Span consumer plugin system with `@register_span_consumer` decorator
- [ ] ~100 lines of source code
- [ ] Tests cover: propagation headers, consumer registration
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/trace/baggage/` (~195 LOC)

---

### US-058: Prompt execution logger

**Description:** As a framework developer, I want structured LLM execution logging with token breakdown and context window usage analysis.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-trace/src/orbiter/trace/prompt_logger.py`
- [ ] Structured logging: token breakdown by role, context window usage
- [ ] Multi-modal content logging (text, images, tool_use)
- [ ] ~130 lines of source code
- [ ] Tests cover: log formatting, token breakdown
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/logs/prompt_log.py` (873 LOC)

---

### Phase 13: Evaluation Package

---

### US-059: Eval types + base evaluator

**Description:** As a framework developer, I want an Evaluator with parallel execution and pass@k metrics so that agent quality can be measured systematically.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-eval/src/orbiter/eval/base.py`
- [ ] `EvalTarget` ABC, `EvalCriteria` with threshold-based pass/fail
- [ ] `Evaluator` with parallel execution, `repeat_times`, pass@k metrics
- [ ] `EvalResult`, `EvalCaseResult`, `ScorerResult` dataclasses
- [ ] ~100 lines of source code
- [ ] Tests cover: evaluator lifecycle, pass@k calculation
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/evaluations/base.py` (~482 LOC)

---

### US-060: Rule-based scorers

**Description:** As a framework user, I want rule-based scorers for format validation, schema compliance, and output correctness checks.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-eval/src/orbiter/eval/scorers.py`
- [ ] `FormatValidationScorer` — JSON, XML, YAML, Markdown, CSV
- [ ] `SchemaValidationScorer` — JSON schema compliance
- [ ] `OutputCorrectnessScorer` — ground truth matching, keyword checking
- [ ] `OutputLengthScorer`, `OutputRelevanceScorer`, `OutputCompletenessScorer`
- [ ] ~130 lines of source code
- [ ] Tests cover: each scorer with pass/fail cases
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/evaluations/scorers/` (~1,200 LOC)

---

### US-061: LLM-as-Judge + quality scorers

**Description:** As a framework user, I want LLM-based scoring with configurable judge prompts and weighted multi-dimensional quality assessment.

**Acceptance Criteria:**
- [ ] Add to `scorers.py` or create `packages/orbiter-eval/src/orbiter/eval/llm_scorer.py`
- [ ] `LLMAsJudgeScorer` — configurable judge prompts
- [ ] `OutputQualityScorer` — weighted 5-dimensional: correctness (40%), relevance (20%), completeness (20%), clarity (10%), professionalism (10%)
- [ ] `LogicConsistencyScorer`, `ReasoningValidityScorer`, `ConstraintSatisfactionScorer`
- [ ] ~130 lines of source code
- [ ] Tests with mocked LLM judge
- [ ] `uv run pytest` — all tests pass

---

### US-062: Trajectory + time scorers

**Description:** As a framework developer, I want trajectory validation, time cost, and accuracy scorers with a scorer registry.

**Acceptance Criteria:**
- [ ] `TrajectoryValidators` — trajectory step validation
- [ ] `TimeCostScorer`, `AnswerAccuracyLLMScorer`, `LabelDistributionScorer`
- [ ] Scorer registry with `@scorer_register()` decorator
- [ ] ~100 lines of source code
- [ ] Tests cover: each scorer, registry decorator
- [ ] `uv run pytest` — all tests pass

---

### US-063: Reflection framework

**Description:** As a framework developer, I want a reflection framework with LLM-powered analysis, insight extraction, and suggestion generation, so that agents can learn from their executions.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-eval/src/orbiter/eval/reflection.py`
- [ ] `Reflector` ABC with three-step template: `analyze()`, `insight()`, `suggest()`
- [ ] `GeneralReflector` using LLM: summary, key_findings, root_causes, insights, suggestions
- [ ] `ReflectionType` enum (SUCCESS, FAILURE, OPTIMIZATION, PATTERN, INSIGHT)
- [ ] `ReflectionLevel` enum (SHALLOW, MEDIUM, DEEP, META)
- [ ] `ReflectionHistory` tracking with summarization
- [ ] ~130 lines of source code
- [ ] Tests with mocked LLM reflection
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/evaluations/reflect/` (~474 LOC)

---

### Phase 13.5: Ralph Loop — Iterative Refinement

---

### US-064: Ralph loop — state + config

**Description:** As a framework developer, I want Ralph loop configuration (validation, reflection, stop conditions) and iteration state tracking, so that iterative refinement has a structured lifecycle.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-eval/src/orbiter/eval/ralph/config.py`
- [ ] `RalphConfig` unifying: `ValidationConfig`, `ReflectionConfig`, `StopConditionConfig`
- [ ] `LoopState` — iteration tracking, score history, reflection history
- [ ] ~120 lines of source code
- [ ] Tests cover: config creation, state transitions
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/ralph_loop/config.py` (119 LOC)

---

### US-065: Ralph loop — stop detectors

**Description:** As a framework developer, I want pluggable stop condition detectors for the Ralph loop (max iterations, timeout, cost limit, score threshold).

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-eval/src/orbiter/eval/ralph/detectors.py`
- [ ] `StopDetector` ABC with pluggable implementations
- [ ] Built-in: `MaxIterationDetector`, `TimeoutDetector`, `CostLimitDetector`, `ConsecutiveFailureDetector`, `ScoreThresholdDetector`
- [ ] ~100 lines of source code
- [ ] Tests cover: each detector type, composite detection
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/ralph_loop/detect/` (~200 LOC)

---

### US-066: Ralph loop — runner

**Description:** As a framework user, I want a RalphRunner that implements the 5-phase Run→Analyze→Learn→Plan→Halt loop for iterative agent refinement.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-eval/src/orbiter/eval/ralph/runner.py`
- [ ] `RalphRunner` — Run → Analyze (score) → Learn (reflect) → Plan (re-prompt) → Halt (detect stop)
- [ ] Integration with Evaluator scorers and Reflection framework
- [ ] ~130 lines of source code
- [ ] Tests cover: full loop with mocked agent, early stopping, score improvement
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/ralph_loop/ralph_runner.py` (425 LOC)

---

### Phase 14: A2A Protocol

---

### US-067: A2A types + agent card

**Description:** As a framework developer, I want A2A protocol types including AgentCard with skills and transport capabilities.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-a2a/src/orbiter/a2a/types.py`
- [ ] `AgentCard` with skills, transport modes, streaming capabilities
- [ ] `ServingConfig`, `ClientConfig` Pydantic models
- [ ] Task event types: `TaskArtifactUpdateEvent`, `TaskStatusUpdateEvent`
- [ ] ~130 lines of source code
- [ ] Tests cover: type creation, serialization
- [ ] `uv run pytest` — all tests pass

---

### US-068: A2A server

**Description:** As a framework user, I want a FastAPI-based A2A server that exposes agents via the A2A protocol with agent card discovery.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-a2a/src/orbiter/a2a/server.py`
- [ ] FastAPI server with `.well-known/agent-card` endpoint
- [ ] Agent executor wrapping, streaming support
- [ ] `TaskStore` abstraction (in-memory default)
- [ ] ~130 lines of source code
- [ ] Tests cover: agent card endpoint, task execution
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/experimental/a2a/agent_server.py` (220 LOC)

---

### US-069: A2A client + remote agent

**Description:** As a framework user, I want an A2A client and RemoteAgent that calls remote A2A agents, so that agents can communicate across services.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-a2a/src/orbiter/a2a/client.py`
- [ ] Thread-safe client manager with per-thread instances
- [ ] Agent card resolution from URL/file
- [ ] `RemoteAgent` — Agent subclass for calling remote A2A agents
- [ ] Task→A2A message conversion, streaming event handling
- [ ] ~130 lines of source code
- [ ] Tests cover: client lifecycle, remote agent invocation (mocked)
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/experimental/a2a/client_proxy.py` (217 LOC), `old/aworld/experimental/a2a/remote_agent.py` (129 LOC)

---

### Phase 15: CLI Package

---

### US-070: CLI entry point + config

**Description:** As a framework user, I want an `orbiter` CLI command with config file support.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-cli/src/orbiter_cli/main.py`
- [ ] CLI entry point with argument parsing
- [ ] Config file loading (`.orbiter.yaml`, `orbiter.config.yaml`)
- [ ] ~130 lines of source code
- [ ] Tests cover: arg parsing, config loading
- [ ] `uv run pytest` — all tests pass

---

### US-071: Agent discovery + loading

**Description:** As a CLI user, I want agents to be auto-discovered from the current directory and loaded for execution.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-cli/src/orbiter_cli/loader.py`
- [ ] Agent file scanning (`.py`, `.yaml`, `.md` agent definitions)
- [ ] Agent registration and validation
- [ ] ~130 lines of source code
- [ ] Tests cover: discovery, loading, validation
- [ ] `uv run pytest` — all tests pass

---

### US-072: Interactive console

**Description:** As a CLI user, I want an interactive REPL console for chatting with agents.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-cli/src/orbiter_cli/console.py`
- [ ] Interactive REPL with prompt, streaming output, command handling
- [ ] ~130 lines of source code
- [ ] Tests cover: command parsing, output formatting
- [ ] `uv run pytest` — all tests pass

---

### US-073: Local executor

**Description:** As a CLI user, I want local agent execution with streaming output and error handling.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-cli/src/orbiter_cli/executor.py`
- [ ] Local execution wrapping `run()` / `run.stream()`
- [ ] ~120 lines of source code
- [ ] Tests cover: execution lifecycle
- [ ] `uv run pytest` — all tests pass

---

### US-074: Plugin system

**Description:** As a CLI user, I want a plugin system for extending CLI functionality.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-cli/src/orbiter_cli/plugins.py`
- [ ] Plugin discovery, loading, lifecycle
- [ ] ~100 lines of source code
- [ ] Tests cover: plugin loading, hook points
- [ ] `uv run pytest` — all tests pass

---

### US-075: Batch execution

**Description:** As a CLI user, I want batch execution mode for running agents against multiple inputs from files or stdin.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-cli/src/orbiter_cli/batch.py`
- [ ] Batch input loading (JSON, CSV, JSONL)
- [ ] Parallel execution with concurrency control
- [ ] Result aggregation and output
- [ ] ~120 lines of source code
- [ ] Tests cover: batch loading, parallel execution
- [ ] `uv run pytest` — all tests pass

---

### Phase 16: Server Package

---

### US-076: FastAPI app + chat route

**Description:** As a framework user, I want a web server with a chat API endpoint for running agents via HTTP.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-server/src/orbiter_server/app.py`
- [ ] FastAPI app with `/chat` endpoint
- [ ] Request/response models, streaming SSE support
- [ ] ~130 lines of source code
- [ ] Tests cover: chat endpoint, streaming
- [ ] `uv run pytest` — all tests pass

---

### US-077: Session management route

**Description:** As a server user, I want session management APIs for creating, listing, and resuming chat sessions.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-server/src/orbiter_server/sessions.py`
- [ ] CRUD routes for sessions
- [ ] ~120 lines of source code
- [ ] Tests cover: session lifecycle
- [ ] `uv run pytest` — all tests pass

---

### US-078: Agent management + workspace routes

**Description:** As a server user, I want API routes for listing agents and accessing workspace artifacts.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-server/src/orbiter_server/agents.py`
- [ ] Agent listing, workspace file listing/reading
- [ ] ~120 lines of source code
- [ ] Tests cover: agent listing, workspace access
- [ ] `uv run pytest` — all tests pass

---

### US-079: Streaming + WebSocket support

**Description:** As a server user, I want WebSocket support for real-time streaming agent output.

**Acceptance Criteria:**
- [ ] WebSocket endpoint for streaming
- [ ] SSE fallback for non-WebSocket clients
- [ ] ~100 lines of source code
- [ ] Tests cover: WebSocket connection, streaming events
- [ ] `uv run pytest` — all tests pass

---

### Phase 17: Training Package

---

### US-080: Trajectory dataset

**Description:** As a training user, I want trajectory capture with strategy patterns and export to JSON/CSV.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-train/src/orbiter/train/trajectory.py`
- [ ] `TrajectoryItem` model, `TrajectoryDataset` with strategy pattern
- [ ] `append_trajectory()`, `from_messages()`, `save_task_trajectory()`
- [ ] Export to JSON/CSV
- [ ] ~130 lines of source code
- [ ] Tests cover: trajectory capture, export
- [ ] `uv run pytest` — all tests pass

**Reference:** `old/aworld/dataset/trajectory_dataset.py` (502 LOC)

---

### US-081: Base trainer

**Description:** As a training user, I want a base Trainer abstraction for fine-tuning agents.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-train/src/orbiter/train/trainer.py`
- [ ] `Trainer` ABC with train/eval lifecycle
- [ ] ~120 lines of source code
- [ ] Tests cover: trainer lifecycle
- [ ] `uv run pytest` — all tests pass

---

### US-082: Data synthesis

**Description:** As a training user, I want data synthesis utilities for generating training data from agent executions.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-train/src/orbiter/train/synthesis.py`
- [ ] Data synthesis from trajectories
- [ ] ~130 lines of source code
- [ ] Tests cover: synthesis pipeline
- [ ] `uv run pytest` — all tests pass

---

### US-083: Agent evolution

**Description:** As a training user, I want agent evolution utilities for iterative improvement.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-train/src/orbiter/train/evolution.py`
- [ ] Agent evolution strategies
- [ ] ~120 lines of source code
- [ ] Tests cover: evolution lifecycle
- [ ] `uv run pytest` — all tests pass

---

### US-084: VeRL integration

**Description:** As a training user, I want VeRL integration for reinforcement learning from human feedback.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-train/src/orbiter/train/verl.py`
- [ ] VeRL integration layer
- [ ] ~130 lines of source code
- [ ] Tests cover: integration (mocked)
- [ ] `uv run pytest` — all tests pass

---

### Phase 18: Examples (Quick-Start)

---

### US-085: Basic examples — define_agent, use_llm

**Description:** As a new user, I want simple examples showing how to define an agent and use LLM providers.

**Acceptance Criteria:**
- [ ] Create `examples/quickstart/define_agent.py`
- [ ] Create `examples/quickstart/use_llm.py`
- [ ] ~80 lines total
- [ ] Examples run without error (with valid API keys)

---

### US-086: Tool examples — local_tool, mcp_tool

**Description:** As a new user, I want examples showing how to create local tools and use MCP tools.

**Acceptance Criteria:**
- [ ] Create `examples/quickstart/local_tool.py`
- [ ] Create `examples/quickstart/mcp_tool.py`
- [ ] ~80 lines total

---

### US-087: Swarm examples — workflow, handoff, hybrid_swarm

**Description:** As a new user, I want examples showing multi-agent orchestration patterns.

**Acceptance Criteria:**
- [ ] Create `examples/quickstart/workflow.py`
- [ ] Create `examples/quickstart/handoff.py`
- [ ] Create `examples/quickstart/hybrid_swarm.py`
- [ ] ~80 lines total

---

### US-088: Memory + trace examples

**Description:** As a new user, I want examples showing memory and tracing usage.

**Acceptance Criteria:**
- [ ] Create `examples/quickstart/use_memory.py`
- [ ] Create `examples/quickstart/use_trace.py`
- [ ] ~80 lines total

---

### US-089: Advanced quickstart — parallel_task, HITL, multi_root_agent, serving

**Description:** As a user, I want examples showing advanced patterns like parallel tasks, human-in-the-loop, multi-root agents, and serving.

**Acceptance Criteria:**
- [ ] Create `examples/quickstart/parallel_task.py`
- [ ] Create `examples/quickstart/hitl.py`
- [ ] Create `examples/quickstart/multi_root_agent.py`
- [ ] Create `examples/quickstart/serving.py`
- [ ] ~100 lines total

---

### US-090: Config-driven + CLI examples

**Description:** As a user, I want examples showing YAML-based agent definitions and CLI usage.

**Acceptance Criteria:**
- [ ] Create `examples/quickstart/config_driven.py`
- [ ] Create `examples/quickstart/agents.yaml`
- [ ] Create `examples/quickstart/cli_usage.py`
- [ ] ~80 lines total

---

### Phase 19: Examples (Multi-Agent Patterns)

---

### US-091: Collaborative examples — debate, travel

**Description:** As a user, I want examples showing collaborative multi-agent patterns.

**Acceptance Criteria:**
- [ ] Create `examples/multi_agent/debate.py`
- [ ] Create `examples/multi_agent/travel.py`
- [ ] ~100 lines total

---

### US-092: Coordination examples — custom_agent, deepresearch, master_worker

**Description:** As a user, I want examples showing coordination patterns.

**Acceptance Criteria:**
- [ ] Create `examples/multi_agent/custom_agent.py`
- [ ] Create `examples/multi_agent/deepresearch.py`
- [ ] Create `examples/multi_agent/master_worker.py`
- [ ] ~100 lines total

---

### US-093: Workflow examples — search patterns

**Description:** As a user, I want examples showing workflow search patterns.

**Acceptance Criteria:**
- [ ] Create `examples/multi_agent/search_patterns.py`
- [ ] ~60 lines total

---

### Phase 20: Examples (Benchmarks)

---

### US-094: GAIA benchmark

**Acceptance Criteria:**
- [ ] Create `examples/benchmarks/gaia/` with benchmark runner
- [ ] ~150 lines

---

### US-095: IMO benchmark

**Acceptance Criteria:**
- [ ] Create `examples/benchmarks/imo/` with benchmark runner
- [ ] ~150 lines

---

### US-096: OSWorld benchmark

**Acceptance Criteria:**
- [ ] Create `examples/benchmarks/osworld/` with benchmark runner
- [ ] ~150 lines

---

### US-097: VisualWebArena benchmark

**Acceptance Criteria:**
- [ ] Create `examples/benchmarks/visualwebarena/` with benchmark runner
- [ ] ~150 lines

---

### US-098: XBench benchmark

**Acceptance Criteria:**
- [ ] Create `examples/benchmarks/xbench/` with benchmark runner
- [ ] ~150 lines

---

### US-099: BFCL + other benchmarks

**Acceptance Criteria:**
- [ ] Create `examples/benchmarks/bfcl/` with benchmark runner
- [ ] ~120 lines

---

### Phase 21: Examples (Advanced)

---

### US-100: Skill agent + web deployment examples

**Acceptance Criteria:**
- [ ] Create `examples/advanced/skill_agent.py`
- [ ] Create `examples/advanced/web_deploy.py`
- [ ] ~120 lines total

---

### US-101: Common tools library

**Acceptance Criteria:**
- [ ] Create `examples/common/tools/` with reusable tool implementations
- [ ] Browser, search API, document tools
- [ ] ~150 lines total

---

### US-102: Training examples

**Acceptance Criteria:**
- [ ] Create `examples/training/` with training pipeline examples
- [ ] ~120 lines total

---

### Phase 22: Final Cleanup & CI

---

### US-103: Public API finalization

**Description:** As a framework developer, I want all package `__init__.py` files to export the correct public API surface with `__all__` declarations.

**Acceptance Criteria:**
- [ ] Audit all `__init__.py` files across all 13 packages
- [ ] Ensure exports match design spec section 1.6
- [ ] Add `__all__` where needed
- [ ] ~50 lines of changes
- [ ] `uv run pytest` — all tests pass

---

### US-104: GitHub Actions CI

**Description:** As a framework developer, I want GitHub Actions CI that runs lint, format, typecheck, and tests on every PR.

**Acceptance Criteria:**
- [ ] Create `.github/workflows/ci.yml`
- [ ] Jobs: ruff check, ruff format, pyright, pytest
- [ ] Matrix: Python 3.11
- [ ] Cache UV dependencies
- [ ] ~80 lines
- [ ] CI passes on push

---

### US-105: README + migration guide

**Description:** As an AWorld user, I want a README and migration guide so that I can understand Orbiter and migrate from AWorld.

**Acceptance Criteria:**
- [ ] Create `README.md` — project overview, installation, quick start, API examples
- [ ] Create `docs/migration-guide.md` — AWorld → Orbiter mapping
- [ ] ~150 lines total

---

### US-106: Delete old/ + final verification

**Description:** As a framework developer, I want to remove the `old/` reference directory and run final verification that everything works without it.

**Acceptance Criteria:**
- [ ] Delete `old/` directory
- [ ] `uv sync` succeeds
- [ ] `uv run ruff check packages/` — zero lint errors
- [ ] `uv run ruff format --check packages/` — all formatted
- [ ] `uv run pyright packages/` — zero type errors
- [ ] `uv run pytest` — all tests pass
- [ ] All examples run (with valid API keys)

---

## Functional Requirements

- FR-1: Agent class must accept `name`, `model`, `instructions`, `tools`, `handoffs`, `hooks`, `output_type`, `max_steps`, `temperature`, `max_tokens`, `memory`, `context` as constructor parameters
- FR-2: `run(agent, input)` must return a `RunResult` with `output`, `messages`, `usage`, `steps`
- FR-3: `run.sync()` must bridge async→sync using `asyncio.run()`
- FR-4: `run.stream()` must yield `TextEvent` and `ToolCallEvent` in real-time
- FR-5: Agent tool loop must execute tool calls, feed results back, and re-call LLM up to `max_steps`
- FR-6: Parallel tool execution must use `asyncio.TaskGroup` when multiple tool calls are returned
- FR-7: Swarm must support three modes: `workflow`, `handoff`, `team`
- FR-8: Flow DSL must parse `"a >> b >> c"` and `"(a | b) >> c"` syntax
- FR-9: Context engine must support hierarchical state with parent inheritance
- FR-10: PromptBuilder must compose neurons in priority order
- FR-11: ContextProcessor pipeline must fire at `pre_llm_call`, `post_llm_call`, `pre_tool_call`, `post_tool_call` events
- FR-12: Workspace must support artifact versioning with `version_history()` and `revert_to_version()`
- FR-13: Checkpoint must save and restore complete execution state
- FR-14: Memory package must support short-term (windowed), summary (compressed), and long-term (extracted) memory
- FR-15: MCP client must support SSE, stdio, and streamable-http transports
- FR-16: YAML agent loader must support `${ENV_VAR}` and `${vars.KEY}` substitution
- FR-17: Skill registry must load from local paths and GitHub URLs with caching
- FR-18: Sandbox tools must validate paths against `allowed_directories`
- FR-19: Trace decorator must support sync, async, generators, and async generators
- FR-20: Ralph loop must implement Run→Analyze→Learn→Plan→Halt 5-phase cycle
- FR-21: A2A server must expose `.well-known/agent-card` endpoint
- FR-22: CLI must support interactive REPL and batch execution modes
- FR-23: All packages must have zero heavy dependencies except their declared extras
- FR-24: `orbiter-core` must depend only on `pydantic` (no provider SDKs)

## Non-Goals (Out of Scope)

- No web UI frontend implementation (server provides API only; web UI is a separate project)
- No CAST, PTC, or continuous modules (explicitly dropped from AWorld)
- No Swift, AREAL, or TRL training integrations (only core + VeRL)
- No automatic migration tool from AWorld to Orbiter (migration guide only)
- No backwards compatibility with AWorld APIs
- No database migrations or ORM integration (memory backends use raw SQL)
- No Kubernetes operator or Helm charts (just client-side Kubernetes sandbox)
- No authentication/authorization layer in the server package
- No package publishing to PyPI (future work after v1.0)

## Technical Considerations

- **Python 3.11 required** — uses `asyncio.TaskGroup`, `ExceptionGroup`, `tomllib`
- **UV workspace monorepo** — all packages managed via UV with workspace protocol
- **Namespace packages** — `orbiter` namespace shared via `pkgutil.extend_path()`
- **Pydantic v2 only** — frozen models for configs/data, `Field()` for validation
- **Async-first** — all internal functions async; only `run.sync()` bridges to sync
- **Max ~200 lines per source file** — split into `_internal/` if larger
- **pytest-asyncio** with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed
- **Pyright limitation** — can't resolve cross-namespace-package imports from `.pth`-based editable installs; use `# pyright: ignore[reportMissingImports]` in test files
- **Test file naming** — must be unique across all packages (pytest collects all `tests/` dirs)

## Success Metrics

- All 106 user stories completed with passing tests
- Zero lint errors (`ruff check packages/`)
- Zero format issues (`ruff format --check packages/`)
- Zero type errors (`pyright packages/`)
- All tests pass (`uv run pytest`) — target 800+ tests total
- All quickstart examples run successfully with valid API keys
- Code stays under ~200 lines per source file
- `orbiter-core` has zero heavy dependencies (only `pydantic`)

## Open Questions

- Should the server package include a basic web UI, or is API-only sufficient for v1?
- Should benchmark examples include dataset downloading utilities, or expect pre-downloaded data?
- Should the Kubernetes sandbox require `kubernetes` as a hard dependency or optional extra?
- Should the CLI support remote execution via A2A in v1, or just local execution?
- What is the target test coverage percentage? (Currently not enforced)
