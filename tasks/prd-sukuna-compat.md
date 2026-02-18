# PRD: Sukuna Compatibility — Distributed Execution, Rich Streaming & Observability

## Introduction

Sukuna is a consumer-facing Django application that uses AWorld as its core agent execution layer. We're migrating Sukuna to use Orbiter instead. A gap analysis identified three critical missing capabilities: distributed agent execution (Sukuna uses Temporal for durable workflows), rich streaming events (Sukuna's frontend/backend depend heavily on AWorld's granular output events), and observability coverage for these new features.

This PRD covers four pillars:
1. **`orbiter-distributed`** — A new package for distributed agent execution via Redis + Temporal
2. **Rich Streaming Events** — AWorld-compatible event types in `orbiter-core`
3. **Observability Addons** — Metrics, tracing, health checks for distributed + streaming
4. **Documentation** — Comprehensive setup and usage guides

### Design Principles
- **Less is more**: Distributed execution should feel like a natural extension of `run()`, not a new paradigm
- **Start basic, design for full durability**: Basic retry and at-least-once delivery first, with architecture that supports exactly-once semantics later
- **Backward compatible**: Existing `TextEvent` and `ToolCallEvent` remain unchanged; new events are additive

## Goals

- Enable Sukuna to replace AWorld with Orbiter without changing its execution model
- Provide distributed agent execution with `run.distributed()` that behaves like local `run()` but on distributed scale
- Match AWorld's streaming event granularity so Sukuna's frontend SSE pipeline works unchanged
- Give operators visibility into distributed worker health, task queues, and streaming performance
- Provide documentation that gets a developer from zero to distributed agents in 5 minutes

## User Stories

---

### PILLAR 1: Rich Streaming Events (orbiter-core)

---

### US-001: Add StepEvent type
**Description:** As a developer consuming streaming events, I want a `StepEvent` emitted at the start and end of each agent step so that my frontend can show step-by-step progress.

**Acceptance Criteria:**
- [ ] `StepEvent` added to `orbiter/types.py` with fields: `type: Literal["step"]`, `step_number: int`, `agent_name: str`, `status: Literal["started", "completed"]`, `started_at: float`, `completed_at: float | None`, `usage: Usage | None`
- [ ] `StepEvent` is a Pydantic `BaseModel` with `.model_dump()` JSON serialization
- [ ] `StreamEvent` union updated to include `StepEvent`
- [ ] Existing `TextEvent` and `ToolCallEvent` unchanged
- [ ] All existing tests pass
- [ ] Typecheck passes (`pyright`)

---

### US-002: Add ToolResultEvent type
**Description:** As a developer, I want a `ToolResultEvent` emitted after each tool execution so that my frontend can display tool results with success/failure status.

**Acceptance Criteria:**
- [ ] `ToolResultEvent` added to `orbiter/types.py` with fields: `type: Literal["tool_result"]`, `tool_name: str`, `tool_call_id: str`, `arguments: dict[str, Any]`, `result: str`, `error: str | None`, `success: bool`, `duration_ms: float`, `agent_name: str`
- [ ] `StreamEvent` union updated to include `ToolResultEvent`
- [ ] Serializable to JSON via `.model_dump()`
- [ ] Typecheck passes

---

### US-003: Add ReasoningEvent, ErrorEvent, StatusEvent, UsageEvent types
**Description:** As a developer, I want additional event types for reasoning content, errors, status changes, and per-step usage so that my frontend has full visibility into agent execution.

**Acceptance Criteria:**
- [ ] `ReasoningEvent` with fields: `type: Literal["reasoning"]`, `text: str`, `agent_name: str`
- [ ] `ErrorEvent` with fields: `type: Literal["error"]`, `error: str`, `error_type: str`, `agent_name: str`, `step_number: int | None`, `recoverable: bool`
- [ ] `StatusEvent` with fields: `type: Literal["status"]`, `status: Literal["starting", "running", "waiting_for_tool", "completed", "cancelled", "error"]`, `agent_name: str`, `message: str`
- [ ] `UsageEvent` with fields: `type: Literal["usage"]`, `usage: Usage`, `agent_name: str`, `step_number: int`, `model: str`
- [ ] All four added to `StreamEvent` union
- [ ] Typecheck passes

---

### US-004: Emit rich events from run.stream()
**Description:** As a developer, I want `run.stream()` to accept a `detailed=True` parameter that enables emission of all rich event types during agent execution.

**Acceptance Criteria:**
- [ ] `run.stream()` signature updated: `async def _stream(agent, input, *, messages=None, provider=None, max_steps=None, detailed=False) -> AsyncIterator[StreamEvent]`
- [ ] When `detailed=False` (default): only `TextEvent` and `ToolCallEvent` emitted (backward compatible)
- [ ] When `detailed=True`: emits `StatusEvent("starting")` at start, `StepEvent(status="started")` before each LLM call, `UsageEvent` after each LLM call, `ToolResultEvent` after each tool execution, `StepEvent(status="completed")` at step end, `StatusEvent("completed")` at finish
- [ ] `ErrorEvent` emitted on errors regardless of `detailed` flag
- [ ] `ReasoningEvent` emitted when model returns reasoning content (if supported by provider)
- [ ] Tests verify both `detailed=False` and `detailed=True` behavior
- [ ] Typecheck passes

---

### US-005: Emit rich events for Swarm streaming
**Description:** As a developer, I want Swarm execution to also support `detailed=True` streaming with events that identify which sub-agent produced each event.

**Acceptance Criteria:**
- [ ] Swarm gets a `stream()` method (or `run.stream()` detects Swarm like `run()` does) accepting `detailed=True`
- [ ] All events include the correct `agent_name` of the sub-agent that produced them
- [ ] `StatusEvent` emitted for each agent handoff/delegation with the new agent name
- [ ] Workflow, handoff, and team modes all emit rich events
- [ ] Tests verify multi-agent streaming events
- [ ] Typecheck passes

---

### US-006: Event filtering and subscription
**Description:** As a developer, I want to filter which event types I receive from the stream so I can subscribe only to events my application cares about.

**Acceptance Criteria:**
- [ ] `run.stream()` accepts optional `event_types: set[str] | None` parameter
- [ ] When `event_types` is provided, only events whose `type` field matches are yielded
- [ ] When `event_types` is `None` (default), all events pass through (respecting `detailed` flag)
- [ ] Example: `event_types={"text", "tool_result"}` yields only `TextEvent` and `ToolResultEvent`
- [ ] Tests verify filtering works correctly
- [ ] Typecheck passes

---

### PILLAR 2: Distributed Agent Execution (orbiter-distributed)

---

### US-007: Create orbiter-distributed package scaffold
**Description:** As a developer, I need the `orbiter-distributed` package set up in the monorepo with proper namespace packaging and dependencies.

**Acceptance Criteria:**
- [ ] Package at `packages/orbiter-distributed/` with `pyproject.toml`
- [ ] Namespace package: `orbiter.distributed` using `pkgutil.extend_path()`
- [ ] Dependencies: `orbiter-core`, `redis[hiredis]>=5.0`, `temporalio>=1.7`
- [ ] Optional dependencies group `[test]` with `pytest`, `pytest-asyncio`, `fakeredis`
- [ ] Added to root `pyproject.toml` workspace members, dev dependencies, and `tool.uv.sources`
- [ ] `__init__.py` with `__all__` exposing public API
- [ ] Package installs cleanly with `uv sync`
- [ ] Typecheck passes

---

### US-008: Redis task queue — TaskBroker
**Description:** As a developer, I need a Redis-backed task broker that can enqueue agent execution tasks and distribute them to workers.

**Acceptance Criteria:**
- [ ] `TaskBroker` class in `orbiter/distributed/broker.py`
- [ ] Constructor: `TaskBroker(redis_url: str, *, queue_name: str = "orbiter:tasks", max_retries: int = 3)`
- [ ] `async connect()` and `async disconnect()` for lifecycle management
- [ ] `async submit(task: TaskPayload) -> str` enqueues task, returns `task_id`
- [ ] `async claim(worker_id: str, timeout: float = 5.0) -> TaskPayload | None` pops task from queue (blocking with timeout)
- [ ] `async ack(task_id: str) -> None` acknowledges task completion
- [ ] `async nack(task_id: str) -> None` returns task to queue for retry
- [ ] Uses Redis Streams for durable task queue (not plain LIST)
- [ ] Consumer groups for multiple workers
- [ ] Tests with fakeredis
- [ ] Typecheck passes

---

### US-009: TaskPayload and TaskStatus models
**Description:** As a developer, I need data models that define the task payload (what gets queued) and task status (tracking progress).

**Acceptance Criteria:**
- [ ] `TaskPayload` Pydantic model in `orbiter/distributed/models.py` with fields: `task_id: str` (auto-generated UUID), `agent_config: dict[str, Any]` (serialized Agent), `input: str`, `messages: list[dict[str, Any]]`, `model: str | None`, `detailed: bool = False`, `metadata: dict[str, Any]` (user-provided context), `created_at: float`, `timeout_seconds: float = 300.0`
- [ ] `TaskStatus` StrEnum: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`, `RETRYING`
- [ ] `TaskResult` Pydantic model: `task_id: str`, `status: TaskStatus`, `result: dict[str, Any] | None` (serialized RunResult), `error: str | None`, `started_at: float | None`, `completed_at: float | None`, `worker_id: str | None`, `retries: int`
- [ ] All models JSON-serializable via `.model_dump()`
- [ ] Typecheck passes

---

### US-010: Task state store
**Description:** As a developer, I need a Redis-backed store that tracks task status, so callers can query task progress and workers can update state.

**Acceptance Criteria:**
- [ ] `TaskStore` class in `orbiter/distributed/store.py`
- [ ] Constructor: `TaskStore(redis_url: str, *, prefix: str = "orbiter:task:", ttl_seconds: int = 86400)`
- [ ] `async set_status(task_id: str, status: TaskStatus, **kwargs) -> None` updates task state in Redis hash
- [ ] `async get_status(task_id: str) -> TaskResult | None` retrieves current task state
- [ ] `async list_tasks(status: TaskStatus | None = None, limit: int = 100) -> list[TaskResult]` lists tasks, optionally filtered
- [ ] Task state stored as Redis hash with TTL for auto-cleanup
- [ ] Tests with fakeredis
- [ ] Typecheck passes

---

### US-011: Result streaming via Redis Pub/Sub and Streams
**Description:** As a developer, I want real-time streaming of agent events from worker to caller via Redis Pub/Sub (for live events) and Redis Streams (for persistence/replay).

**Acceptance Criteria:**
- [ ] `EventPublisher` class in `orbiter/distributed/events.py`
- [ ] `async publish(task_id: str, event: StreamEvent) -> None` publishes to both Redis Pub/Sub channel (`orbiter:events:{task_id}`) and appends to Redis Stream (`orbiter:stream:{task_id}`)
- [ ] `EventSubscriber` class with two consumption modes:
  - `async subscribe(task_id: str) -> AsyncIterator[StreamEvent]` — live Pub/Sub subscription
  - `async replay(task_id: str, from_id: str = "0") -> AsyncIterator[StreamEvent]` — reads from persistent Stream
- [ ] Events serialized as JSON, deserialized back to proper `StreamEvent` subtype via discriminated union on `type` field
- [ ] Pub/Sub channel auto-cleanup after task completion
- [ ] Stream TTL configurable (default 1 hour)
- [ ] Tests with fakeredis
- [ ] Typecheck passes

---

### US-012: Worker process — task execution loop
**Description:** As a developer, I need a worker process that claims tasks from the queue, executes agents, publishes streaming events, and updates task status.

**Acceptance Criteria:**
- [ ] `Worker` class in `orbiter/distributed/worker.py`
- [ ] Constructor: `Worker(redis_url: str, *, worker_id: str | None = None, concurrency: int = 1, queue_name: str = "orbiter:tasks")`
- [ ] `async start()` enters claim-execute loop; `async stop()` gracefully shuts down
- [ ] On task claim: sets status to `RUNNING`, reconstructs Agent from `agent_config`, calls `run.stream(agent, input, detailed=task.detailed)`, publishes each `StreamEvent` via `EventPublisher`, on completion sets status to `COMPLETED` with `RunResult`
- [ ] On failure: sets status to `FAILED` with error, `nack`s task if retries remain
- [ ] Worker generates unique `worker_id` if not provided (hostname + PID + random suffix)
- [ ] Heartbeat: worker publishes heartbeat to `orbiter:workers:{worker_id}` Redis key with TTL (30s default)
- [ ] Handles `SIGINT`/`SIGTERM` for graceful shutdown
- [ ] Tests verify task execution lifecycle
- [ ] Typecheck passes

---

### US-013: Task cancellation
**Description:** As a developer, I want to cancel a running distributed task so that the worker stops execution and frees resources.

**Acceptance Criteria:**
- [ ] `CancellationToken` class in `orbiter/distributed/cancel.py` with `cancelled: bool` property and `cancel()` method
- [ ] `TaskBroker.cancel(task_id: str) -> None` publishes cancel signal to `orbiter:cancel:{task_id}` Pub/Sub channel and sets task status to `CANCELLED`
- [ ] Worker subscribes to cancel channel for active tasks; on cancel signal, sets `CancellationToken.cancelled = True`
- [ ] Agent execution loop checks `CancellationToken` between steps (cooperative cancellation)
- [ ] `StatusEvent(status="cancelled")` emitted when cancellation takes effect
- [ ] Tests verify cancellation flow
- [ ] Typecheck passes

---

### US-014: run.distributed() API
**Description:** As a developer, I want a `run.distributed()` function that submits agent execution to the distributed queue and returns a handle for monitoring results — behaving like the local `run()` but on distributed scale.

**Acceptance Criteria:**
- [ ] `run.distributed()` function in `orbiter/distributed/client.py` with signature: `async def distributed(agent: Agent | Swarm, input: str, *, messages=None, redis_url: str | None = None, detailed: bool = False, timeout: float = 300.0, metadata: dict | None = None) -> TaskHandle`
- [ ] `TaskHandle` class with methods:
  - `task_id: str` property
  - `async result() -> RunResult` blocks until completion (polls TaskStore)
  - `async stream() -> AsyncIterator[StreamEvent]` subscribes to live events via EventSubscriber
  - `async cancel() -> None` cancels the task
  - `async status() -> TaskResult` returns current task status
- [ ] `redis_url` defaults to `ORBITER_REDIS_URL` environment variable
- [ ] Agent/Swarm serialized to `TaskPayload.agent_config` dict
- [ ] Works with both `Agent` and `Swarm` instances
- [ ] Example usage:
  ```python
  handle = await run.distributed(agent, "Hello", redis_url="redis://localhost")
  async for event in handle.stream():
      print(event)
  result = await handle.result()
  ```
- [ ] Tests verify submit → execute → result flow
- [ ] Typecheck passes

---

### US-015: Agent and Swarm serialization/deserialization
**Description:** As a developer, I need Agent and Swarm to be serializable to dict (for task payloads) and reconstructable from dict (on worker side).

**Acceptance Criteria:**
- [ ] `Agent.to_dict() -> dict[str, Any]` serializes agent config (name, model, instructions, tool references, handoffs, max_steps, temperature, max_tokens, output_type reference)
- [ ] `Agent.from_dict(data: dict) -> Agent` reconstructs agent (tools resolved by name from a registry or via importable references)
- [ ] `Swarm.to_dict() -> dict[str, Any]` serializes swarm config including all agents
- [ ] `Swarm.from_dict(data: dict) -> Swarm` reconstructs swarm
- [ ] Tools serialized as importable dotted paths (e.g., `"myapp.tools.search"`) and resolved via import on deserialization
- [ ] Round-trip test: `Agent.from_dict(agent.to_dict())` produces functionally equivalent agent
- [ ] Typecheck passes

---

### US-016: CLI command — orbiter start worker
**Description:** As an operator, I want to start a distributed worker via CLI so I can scale agent execution horizontally.

**Acceptance Criteria:**
- [ ] `orbiter start worker` command added to `orbiter-cli` via Typer subcommand
- [ ] Options: `--redis-url` (default: `ORBITER_REDIS_URL` env var), `--concurrency` (default: 1), `--queue` (default: `orbiter:tasks`), `--worker-id` (auto-generated if not set)
- [ ] Starts `Worker` instance and runs until `SIGINT`/`SIGTERM`
- [ ] Prints startup banner with worker ID, Redis URL (masked), queue name, concurrency
- [ ] Rich console output showing task processing activity
- [ ] `orbiter-cli` adds `orbiter-distributed` as a dependency
- [ ] Tests verify CLI command registration
- [ ] Typecheck passes

---

### US-017: CLI command — orbiter task status/cancel/list
**Description:** As an operator, I want CLI commands to inspect and manage distributed tasks.

**Acceptance Criteria:**
- [ ] `orbiter task status <task_id>` — shows task status, worker, timing, result preview
- [ ] `orbiter task cancel <task_id>` — cancels a running task
- [ ] `orbiter task list` — lists recent tasks with status, optionally filtered by `--status`
- [ ] Output formatted with Rich tables
- [ ] `--redis-url` option (defaults to `ORBITER_REDIS_URL`)
- [ ] Tests verify CLI commands
- [ ] Typecheck passes

---

### US-018: Temporal integration for durable execution
**Description:** As a developer, I want optional Temporal integration for durable workflow execution, so tasks survive worker crashes and can be retried with full state recovery.

**Acceptance Criteria:**
- [ ] `TemporalExecutor` class in `orbiter/distributed/temporal.py` as an alternative execution backend
- [ ] Temporal workflow wraps agent execution: receives `TaskPayload`, executes agent in an activity, publishes events
- [ ] Temporal activity: runs `run.stream()` with heartbeating (Temporal activity heartbeat)
- [ ] `Worker` accepts `executor: Literal["local", "temporal"] = "local"` parameter
- [ ] When `executor="temporal"`: worker registers Temporal workflows/activities instead of direct execution
- [ ] Temporal connection configured via `TEMPORAL_HOST` and `TEMPORAL_NAMESPACE` env vars
- [ ] `temporalio` is an optional dependency (not required for Redis-only mode)
- [ ] Graceful degradation: if `temporalio` not installed and `executor="temporal"` requested, raise clear error
- [ ] Tests for Temporal workflow/activity logic (mocked Temporal client)
- [ ] Typecheck passes

---

### US-019: Worker health checks and monitoring endpoint
**Description:** As an operator, I want workers to expose health information so I can monitor fleet health.

**Acceptance Criteria:**
- [ ] Worker publishes health data to `orbiter:workers:{worker_id}` Redis hash: `status`, `tasks_processed`, `tasks_failed`, `current_task_id`, `started_at`, `last_heartbeat`, `concurrency`, `hostname`
- [ ] `WorkerHealthCheck` class implementing `orbiter.observability.health.HealthCheck` protocol
- [ ] `async get_worker_fleet_status(redis_url: str) -> list[WorkerHealth]` function returns all active workers
- [ ] Workers with expired heartbeat TTL (>60s) considered dead
- [ ] CLI: `orbiter worker list` shows all workers and their health
- [ ] Tests verify health data publishing
- [ ] Typecheck passes

---

### PILLAR 3: Observability Addons

---

### US-020: Distributed task metrics
**Description:** As an operator, I want metrics for distributed task processing so I can monitor queue health and worker performance.

**Acceptance Criteria:**
- [ ] New metric constants in `orbiter/observability/semconv.py`: `METRIC_DIST_TASKS_SUBMITTED`, `METRIC_DIST_TASKS_COMPLETED`, `METRIC_DIST_TASKS_FAILED`, `METRIC_DIST_TASKS_CANCELLED`, `METRIC_DIST_QUEUE_DEPTH`, `METRIC_DIST_TASK_DURATION`, `METRIC_DIST_TASK_WAIT_TIME`
- [ ] New semantic convention attributes: `DIST_TASK_ID`, `DIST_WORKER_ID`, `DIST_QUEUE_NAME`, `DIST_TASK_STATUS`
- [ ] `record_task_submitted()`, `record_task_completed()`, `record_task_failed()` helper functions in `orbiter/distributed/metrics.py`
- [ ] Metrics recorded using existing `orbiter.observability.metrics` infrastructure (both OTel and in-memory fallback)
- [ ] Worker automatically records metrics during task lifecycle
- [ ] Tests verify metrics are recorded
- [ ] Typecheck passes

---

### US-021: Distributed task tracing
**Description:** As an operator, I want distributed traces that span from task submission through queue to worker execution so I can trace requests end-to-end.

**Acceptance Criteria:**
- [ ] Task submission creates a trace span `orbiter.distributed.submit` with task_id, agent_name attributes
- [ ] Trace context propagated in `TaskPayload.metadata["trace_context"]` using W3C baggage from `orbiter.observability.propagation`
- [ ] Worker extracts trace context and creates child span `orbiter.distributed.execute` linked to submission span
- [ ] Tool executions within worker create nested spans (existing `@traced` decorator)
- [ ] Uses existing `orbiter.observability.tracing` infrastructure (`aspan`, `traced`)
- [ ] Tests verify trace context propagation
- [ ] Typecheck passes

---

### US-022: Streaming event metrics
**Description:** As an operator, I want metrics on streaming events so I can monitor event throughput and latency.

**Acceptance Criteria:**
- [ ] New metric constants: `METRIC_STREAM_EVENTS_EMITTED`, `METRIC_STREAM_EVENT_PUBLISH_DURATION`
- [ ] New attribute: `STREAM_EVENT_TYPE` for event type breakdown
- [ ] `EventPublisher` records metrics on each publish (event count by type, publish duration)
- [ ] `run.stream()` with `detailed=True` records total events emitted per run
- [ ] Metrics recorded using existing infrastructure
- [ ] Tests verify event metrics
- [ ] Typecheck passes

---

### US-023: Distributed alert rules
**Description:** As an operator, I want pre-defined alert rules for distributed system health so I get notified of problems.

**Acceptance Criteria:**
- [ ] `register_distributed_alerts()` function in `orbiter/distributed/alerts.py`
- [ ] Pre-defined alert rules:
  - Queue depth > 100 tasks (WARNING)
  - Queue depth > 500 tasks (CRITICAL)
  - Task failure rate > 10% (WARNING)
  - Worker count = 0 (CRITICAL)
  - Task wait time > 60s (WARNING)
- [ ] Uses existing `orbiter.observability.alerts.AlertManager` and `AlertRule`
- [ ] Alert rules registered with global alert manager
- [ ] Tests verify alert rules trigger correctly
- [ ] Typecheck passes

---

### PILLAR 4: Documentation

---

### US-024: Package README for orbiter-distributed
**Description:** As a developer, I want a README in the orbiter-distributed package that explains what it does and shows basic usage.

**Acceptance Criteria:**
- [ ] `packages/orbiter-distributed/README.md` with: package description, installation, quick start (5-minute guide), basic `run.distributed()` example, worker startup command, link to full docs
- [ ] Covers both Agent and Swarm usage
- [ ] Shows environment variable configuration

---

### US-025: Architecture documentation
**Description:** As a developer, I want architecture docs explaining how distributed execution components connect.

**Acceptance Criteria:**
- [ ] `docs/distributed/architecture.md` with: system architecture diagram (ASCII or Mermaid), component descriptions (TaskBroker, Worker, EventPublisher, TaskStore), data flow (submit → queue → worker → events → caller), Redis data structures used, Temporal integration explanation
- [ ] Covers both Redis-only and Redis+Temporal modes

---

### US-026: Worker setup and scaling guide
**Description:** As an operator, I want a guide on configuring and scaling workers.

**Acceptance Criteria:**
- [ ] `docs/distributed/workers.md` with: worker startup options, concurrency tuning, multiple worker deployment, environment variables reference, Redis configuration recommendations, health monitoring setup
- [ ] Includes Docker and docker-compose examples for worker deployment

---

### US-027: Migration guide — local to distributed
**Description:** As a developer, I want a guide showing how to move from local `run()` to distributed `run.distributed()`.

**Acceptance Criteria:**
- [ ] `docs/distributed/migration.md` with: side-by-side comparison of local vs distributed code, step-by-step migration walkthrough, handling streaming events in both modes, error handling differences, when to use local vs distributed

---

### US-028: Rich streaming events documentation
**Description:** As a developer, I want documentation for all streaming event types and how to use them.

**Acceptance Criteria:**
- [ ] `docs/streaming-events.md` with: all event types with field descriptions, `detailed=True` usage guide, event filtering examples, SSE integration example (FastAPI/Django), frontend consumption patterns, migration from AWorld output types

---

### US-029: Example applications
**Description:** As a developer, I want example applications demonstrating distributed execution.

**Acceptance Criteria:**
- [ ] `examples/distributed/simple_chat.py` — simple chatbot agent running distributed with result streaming
- [ ] `examples/distributed/multi_agent.py` — Swarm with multiple agents running distributed
- [ ] `examples/distributed/docker-compose.yml` — Docker Compose with Redis, Temporal (optional), and 2 workers
- [ ] `examples/distributed/fastapi_sse.py` — FastAPI endpoint that submits distributed task and streams SSE events to frontend
- [ ] Each example has inline comments explaining the code
- [ ] Examples run successfully with `uv run`

---

## Functional Requirements

- FR-1: `StreamEvent` union must include all 8 event types: `TextEvent`, `ToolCallEvent`, `StepEvent`, `ToolResultEvent`, `ReasoningEvent`, `ErrorEvent`, `StatusEvent`, `UsageEvent`
- FR-2: `run.stream(detailed=True)` emits rich events; `detailed=False` (default) emits only `TextEvent` and `ToolCallEvent`
- FR-3: `run.distributed()` submits task to Redis queue and returns `TaskHandle`
- FR-4: `TaskHandle.stream()` yields real-time `StreamEvent` objects from worker via Redis Pub/Sub
- FR-5: `TaskHandle.result()` blocks until task completion and returns `RunResult`
- FR-6: `TaskHandle.cancel()` sends cooperative cancellation signal to worker
- FR-7: Worker claims tasks from Redis Stream consumer group, executes agents, publishes events
- FR-8: Worker heartbeat via Redis key with TTL; dead workers detected by expired heartbeat
- FR-9: `orbiter start worker` CLI command starts worker with only `ORBITER_REDIS_URL` required
- FR-10: `orbiter task list/status/cancel` CLI commands for task management
- FR-11: Optional Temporal integration for durable execution (worker restarts resume tasks)
- FR-12: Distributed task metrics, traces, and alerts integrated with `orbiter-observability`
- FR-13: Redis Pub/Sub for real-time events, Redis Streams for persistent/replayable events
- FR-14: Agent and Swarm serializable to dict for task payloads, reconstructable on worker side

## Non-Goals

- No built-in HTTP/REST API server (Sukuna owns its Django layer; FastAPI examples are for reference only)
- No custom frontend or dashboard UI (operators use existing monitoring tools + CLI)
- No multi-tenancy or authentication in the distributed system (Sukuna handles this at application layer)
- No automatic scaling of workers (use external orchestration like Kubernetes HPA)
- No cross-region or multi-datacenter distribution
- No built-in rate limiting (use Redis-based rate limiting at application layer if needed)
- No message priority queues in v1 (all tasks treated equally; priority can be added later)

## Technical Considerations

- **Redis version**: Requires Redis 6.2+ for Stream consumer groups with `XAUTOCLAIM`
- **Temporal**: Optional dependency — package works with Redis-only mode for simpler deployments
- **Serialization**: Agent tools must be importable by dotted path for worker-side reconstruction; closure-based tools won't work distributed (document this clearly)
- **Namespace packaging**: `orbiter.distributed` follows existing `pkgutil.extend_path()` pattern
- **Testing**: Use `fakeredis` for Redis tests, mock Temporal client for Temporal tests
- **Backward compatibility**: All changes to `orbiter-core` are additive; no breaking changes to existing API
- **Provider credentials**: Workers need access to LLM provider API keys (via environment variables) — document this requirement

## Success Metrics

- Developer can go from zero to distributed agent in under 5 minutes with quick start guide
- `run.distributed()` API requires no more than 2 additional lines vs local `run()`
- All AWorld streaming event types have Orbiter equivalents
- Worker fleet can be monitored via CLI and observability metrics
- All quality checks pass (pyright, ruff, pytest)

## Open Questions

- Should `run.distributed()` be attached to the existing `run` function object (like `run.sync` and `run.stream`) or be a separate import from `orbiter.distributed`? (Recommendation: separate import to avoid coupling core to distributed dependency)
- Should workers support loading agent configs from YAML files (like `orbiter-cli` does) in addition to serialized dicts? (Recommendation: yes, as a follow-up story)
- What is the maximum reasonable payload size for Redis Streams? Should large tool results be stored separately with only a reference in the stream? (Recommendation: document 1MB soft limit, add reference-based storage later if needed)
