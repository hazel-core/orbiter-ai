# PRD: orbiter-observability

## Introduction

Consolidate all observability concerns — structured logging, distributed tracing, metrics, prompt logging, health checks, alerting, cost estimation, and SLO tracking — into a single `orbiter-observability` package. This replaces both `orbiter.log` (in orbiter-core) and the standalone `orbiter-trace` package with a unified, production-grade observability layer.

The package follows a **stdlib-first, OTel-optional** design: zero external dependencies for baseline structured logging and metrics collection, with optional OpenTelemetry integration for production export to Jaeger, Datadog, Grafana, etc.

All instrumentation is **explicit opt-in** — nothing happens unless user code calls `configure()` or uses decorators.

## Goals

- Replace `orbiter.log` (87 LOC in orbiter-core) and `orbiter-trace` (~900 LOC, 6 files) with a single consolidated package
- Zero required external dependencies for the baseline (stdlib `logging` only)
- Optional `opentelemetry` extra for production-grade trace/metric export
- Production-grade features: health checks, alerting hooks, SLO tracking, cost estimation
- Follow existing monorepo patterns (namespace package, hatchling, pydantic config)
- Explicit opt-in — no monkey-patching, no auto-instrumentation
- 100% test coverage on all new modules

## User Stories

### US-001: Create package scaffold and configuration
**Description:** As a developer, I need the `orbiter-observability` package set up in the monorepo with its pyproject.toml, namespace package init, and configuration model so that all subsequent modules have a foundation.

**Acceptance Criteria:**
- [ ] `packages/orbiter-observability/` directory created with `pyproject.toml` following existing patterns (hatchling, `src/orbiter/` layout)
- [ ] `src/orbiter/observability/__init__.py` with `pkgutil.extend_path` not needed (sub-namespace under `orbiter.observability`)
- [ ] `ObservabilityConfig` pydantic model in `config.py` with: `log_level` (str, default "WARNING"), `log_format` ("text" | "json", default "text"), `trace_enabled` (bool, default False), `trace_backend` (enum: otlp/memory/console), `trace_endpoint` (str|None), `service_name` (str, default "orbiter"), `sample_rate` (float, 0.0–1.0), `metrics_enabled` (bool, default False), `namespace` (str, default "orbiter")
- [ ] `configure(config: ObservabilityConfig | None = None, **kwargs)` top-level function that initializes all subsystems
- [ ] Package added to root `pyproject.toml` workspace members and dev dependencies
- [ ] `[project.optional-dependencies]` includes `otel = ["opentelemetry-api>=1.20", "opentelemetry-sdk>=1.20"]` and `otlp = ["opentelemetry-exporter-otlp>=1.20"]`
- [ ] Base dependencies: only `pydantic>=2.0` (no opentelemetry in base)
- [ ] `uv sync` succeeds with the new package
- [ ] Tests: config validation, configure() idempotency, kwargs override
- [ ] Pyright passes

### US-002: Structured logging module
**Description:** As a developer, I want a structured logging system that replaces `orbiter.log` so I get consistent, formatted log output across all orbiter packages with zero external dependencies.

**Acceptance Criteria:**
- [ ] `orbiter/observability/logging.py` with `get_logger(name)` and `configure_logging(level, format)` functions
- [ ] Text formatter: compact single-line ANSI output (port from existing `orbiter.log._Formatter`)
- [ ] JSON formatter: structured `{"timestamp", "level", "logger", "message", "extra"}` output for production
- [ ] Format selection via `ObservabilityConfig.log_format` ("text" | "json")
- [ ] Auto-namespace: `get_logger("agent")` → logger named `orbiter.agent`
- [ ] Idempotent configure (calling twice is a no-op unless `force=True`)
- [ ] Context injection: optional `context_vars` dict attached to log records (agent_name, task_id, session_id)
- [ ] `LogContext` context manager that binds key-value pairs to all logs within its scope (via contextvars)
- [ ] Zero external dependencies — stdlib `logging` only
- [ ] Tests: both formatters, namespace auto-prefixing, idempotency, LogContext binding
- [ ] Pyright passes

### US-003: Distributed tracing with optional OTel
**Description:** As a developer, I want a `@traced` decorator and span context managers that work with or without OpenTelemetry installed, so I can instrument code without requiring OTel in development.

**Acceptance Criteria:**
- [ ] `orbiter/observability/tracing.py` with `@traced` decorator, `span()` sync context manager, `aspan()` async context manager
- [ ] When OTel is **not installed**: `@traced` is a no-op passthrough, `span()`/`aspan()` yield a `NullSpan` stub
- [ ] When OTel **is installed**: delegates to real `opentelemetry.trace` spans
- [ ] `@traced` supports: sync functions, async functions, sync generators, async generators
- [ ] Function metadata extraction: qualname, module, lineno, parameters (port from existing `decorator.py`)
- [ ] User-code frame filtering: `is_user_code()`, `get_user_frame()` (port from existing)
- [ ] `extract_args=True` option records call arguments as span attributes
- [ ] Exception recording on spans (set status, record_exception)
- [ ] Tests: all 4 function types with and without OTel, NullSpan behavior, metadata extraction, exception recording
- [ ] Pyright passes

### US-004: Metrics collection
**Description:** As a developer, I want to record agent and tool metrics (counters, histograms) with a stdlib fallback when OTel is not installed.

**Acceptance Criteria:**
- [ ] `orbiter/observability/metrics.py` with metric recording functions
- [ ] `MetricsCollector` class: in-memory metric storage when OTel is not installed
- [ ] Instrument factories: `create_counter(name, unit, description)`, `create_histogram(name, unit, description)`, `create_gauge(name, unit, description)`
- [ ] When OTel is installed: delegates to `opentelemetry.metrics` meter
- [ ] When OTel is not installed: records to in-memory `MetricsCollector` (dict-based)
- [ ] `record_agent_run(duration, success, attributes, input_tokens, output_tokens)` helper (port from existing)
- [ ] `record_tool_step(duration, success, attributes)` helper (port from existing)
- [ ] `Timer` context manager for measuring durations (port from existing)
- [ ] `get_metrics_snapshot() -> dict` for reading in-memory metrics (useful for tests and health checks)
- [ ] Attribute builders: `build_agent_attributes()`, `build_tool_attributes()` (port from existing)
- [ ] Tests: both OTel and stdlib paths, Timer, snapshot, attribute builders
- [ ] Pyright passes

### US-005: Semantic conventions
**Description:** As a developer, I need standardized attribute names for agent, tool, LLM, and task spans/metrics so that all orbiter telemetry is consistent and queryable.

**Acceptance Criteria:**
- [ ] `orbiter/observability/semconv.py` with all semantic convention constants
- [ ] GenAI conventions: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`, etc. (port from existing `config.py`)
- [ ] Agent conventions: `orbiter.agent.{id,name,type,model,step,max_steps,run.success}`
- [ ] Tool conventions: `orbiter.tool.{name,call_id,arguments,result,error,duration,step.success}`
- [ ] Task/session conventions: `orbiter.task.id`, `orbiter.session.id`, `orbiter.user.id`
- [ ] Span name prefixes: `agent.`, `tool.`, `llm.`, `task.`
- [ ] Cost conventions (new): `orbiter.cost.input_tokens`, `orbiter.cost.output_tokens`, `orbiter.cost.total_usd`
- [ ] Tests: constants are strings, no duplicates, naming pattern validation
- [ ] Pyright passes

### US-006: Prompt execution logger
**Description:** As a developer, I want structured logging of LLM executions — token breakdown by role, context window usage, and duration — so I can monitor and debug agent behavior.

**Acceptance Criteria:**
- [ ] `orbiter/observability/prompt_logger.py` with `PromptLogger`, `TokenBreakdown`, `ExecutionLogEntry`
- [ ] `estimate_tokens(text, ratio)` for heuristic token counting (port from existing)
- [ ] `compute_token_breakdown(messages)` with per-role counting: system, user, assistant, tool, other (port from existing)
- [ ] Multi-modal content handling: text items summed, images as fixed estimate, tool_use serialization cost
- [ ] `ExecutionLogEntry` dataclass with `format_summary()` for human-readable output
- [ ] `PromptLogger.log_execution()` computes breakdown + logs structured entry
- [ ] Context window percentage breakdown (used/free with per-role percentages)
- [ ] Tests: token estimation, breakdown computation, multi-modal, format_summary, log_execution
- [ ] Pyright passes

### US-007: Trace context propagation
**Description:** As a developer, I want W3C Baggage propagation and a span consumer plugin system so that trace context flows across service boundaries.

**Acceptance Criteria:**
- [ ] `orbiter/observability/propagation.py` with `BaggagePropagator`, `Carrier`, `DictCarrier`, `SpanConsumer`
- [ ] W3C Baggage extract/inject with URL-encoding and size limits (RFC 9110) — port from existing
- [ ] `Carrier` protocol for header containers + `DictCarrier` implementation
- [ ] Async-safe baggage storage via `contextvars.ContextVar`
- [ ] `set_baggage()`, `get_baggage()`, `get_baggage_value()`, `clear_baggage()` functions
- [ ] `SpanConsumer` ABC with `register_span_consumer()`, `dispatch_spans()`, `list_span_consumers()`, `clear_span_consumers()`
- [ ] Tests: extract/inject round-trip, size limits, URL encoding, baggage context isolation, consumer lifecycle
- [ ] Pyright passes

### US-008: Health check system
**Description:** As an operator, I want health check endpoints so that I can monitor whether the observability pipeline is functioning and whether agents are healthy.

**Acceptance Criteria:**
- [ ] `orbiter/observability/health.py` with `HealthCheck`, `HealthStatus`, `HealthRegistry`
- [ ] `HealthStatus` enum: `HEALTHY`, `DEGRADED`, `UNHEALTHY`
- [ ] `HealthCheck` protocol: `name: str`, `check() -> HealthResult` (with status, message, metadata dict)
- [ ] `HealthResult` dataclass: `status: HealthStatus`, `message: str`, `metadata: dict`, `checked_at: datetime`
- [ ] `HealthRegistry`: register checks, run all checks, return aggregate status
- [ ] Built-in checks: `MemoryUsageCheck` (RSS threshold), `EventLoopCheck` (lag threshold)
- [ ] `get_health_summary() -> dict` returns JSON-serializable health report
- [ ] Aggregate status: UNHEALTHY if any check is UNHEALTHY, DEGRADED if any is DEGRADED, else HEALTHY
- [ ] Tests: registry lifecycle, aggregate logic, built-in checks with mocked thresholds
- [ ] Pyright passes

### US-009: Alerting hooks
**Description:** As a developer, I want to register alerting callbacks that fire when metrics exceed thresholds so that I can integrate with external alerting systems (PagerDuty, Slack, etc.).

**Acceptance Criteria:**
- [ ] `orbiter/observability/alerts.py` with `AlertRule`, `AlertManager`, `AlertSeverity`
- [ ] `AlertSeverity` enum: `INFO`, `WARNING`, `CRITICAL`
- [ ] `AlertRule` dataclass: `name`, `metric_name`, `threshold`, `comparator` (gt/lt/gte/lte/eq), `severity`, `cooldown_seconds` (default 300)
- [ ] `AlertCallback` protocol: `(alert: Alert) -> None` (sync or async)
- [ ] `Alert` dataclass: `rule_name`, `metric_value`, `threshold`, `severity`, `timestamp`, `metadata`
- [ ] `AlertManager`: register rules, register callbacks, `evaluate(metric_name, value)` checks rules and fires callbacks
- [ ] Cooldown enforcement: same rule doesn't fire again within `cooldown_seconds`
- [ ] `clear_alerts()`, `list_rules()` for management
- [ ] Tests: rule evaluation, threshold comparators, cooldown, callback invocation, multiple rules per metric
- [ ] Pyright passes

### US-010: Cost estimation
**Description:** As a developer, I want automatic cost tracking per LLM call so I can monitor spending across agents and models.

**Acceptance Criteria:**
- [ ] `orbiter/observability/cost.py` with `CostTracker`, `CostEntry`, `ModelPricing`
- [ ] `ModelPricing` dataclass: `model_pattern` (regex), `input_cost_per_1k` (float), `output_cost_per_1k` (float)
- [ ] Built-in pricing table for common models: gpt-4o, gpt-4o-mini, claude-sonnet-4-5-20250514, claude-haiku-3-5-20241022, gemini-2.0-flash (updatable via `register_pricing()`)
- [ ] `CostEntry` dataclass: `model`, `input_tokens`, `output_tokens`, `input_cost`, `output_cost`, `total_cost`, `timestamp`
- [ ] `CostTracker`: `record(model, input_tokens, output_tokens) -> CostEntry`, `get_total() -> float`, `get_breakdown() -> dict[str, float]` (per-model), `get_entries(since: datetime | None) -> list[CostEntry]`
- [ ] Thread-safe (uses threading.Lock)
- [ ] `reset()` to clear tracked costs
- [ ] Tests: pricing lookup, cost calculation, breakdown, unknown model handling (returns 0 cost with warning), thread safety
- [ ] Pyright passes

### US-011: SLO tracking
**Description:** As an operator, I want to define Service Level Objectives and track compliance so I can measure reliability of agent operations.

**Acceptance Criteria:**
- [ ] `orbiter/observability/slo.py` with `SLO`, `SLOTracker`, `SLOReport`
- [ ] `SLO` dataclass: `name`, `metric_name`, `target` (float, e.g., 0.99 for 99%), `window_seconds` (default 3600), `comparator` ("lt" for latency, "gt" for success rate)
- [ ] `SLOTracker`: `register(slo)`, `record(metric_name, value)`, `report(slo_name) -> SLOReport`
- [ ] `SLOReport` dataclass: `slo_name`, `target`, `actual`, `budget_remaining` (float 0-1), `compliant` (bool), `window_start`, `total_samples`, `violating_samples`
- [ ] Sliding window: only considers samples within `window_seconds` of now
- [ ] `report_all() -> list[SLOReport]` for dashboard integration
- [ ] `reset()` to clear tracking data
- [ ] Tests: SLO compliance/violation, sliding window expiry, budget calculation, multiple SLOs
- [ ] Pyright passes

### US-012: Update orbiter-core to use orbiter-observability
**Description:** As a developer, I need orbiter-core to depend on orbiter-observability instead of its inline `log.py`, and update all internal imports.

**Acceptance Criteria:**
- [ ] `orbiter-core/pyproject.toml` adds `orbiter-observability>=0.1.0` dependency with `[tool.uv.sources]` workspace entry
- [ ] `orbiter.log` module replaced with a thin re-export shim: `from orbiter.observability.logging import get_logger, configure` (backward compatibility)
- [ ] All internal imports in orbiter-core (`agent.py`, `swarm.py`, `call_runner.py`) updated to use `orbiter.observability.logging`
- [ ] `orbiter/__init__.py` exports `configure`, `get_logger` from new location
- [ ] All existing `test_log.py` tests still pass
- [ ] `uv sync` succeeds
- [ ] Pyright passes

### US-013: Remove orbiter-trace package
**Description:** As a developer, I need to remove the now-redundant `orbiter-trace` package since all its functionality is consolidated into orbiter-observability.

**Acceptance Criteria:**
- [ ] `packages/orbiter-trace/` directory deleted
- [ ] `orbiter-trace` removed from root `pyproject.toml` workspace members and dev dependencies
- [ ] Any packages that depended on `orbiter-trace` updated to depend on `orbiter-observability` instead
- [ ] All existing orbiter-trace tests ported to `packages/orbiter-observability/tests/` and passing
- [ ] `uv sync` succeeds
- [ ] Full test suite passes (`pytest packages/`)
- [ ] Pyright passes

### US-014: Public API and __init__.py exports
**Description:** As a developer, I want a clean public API from `orbiter.observability` that exposes all key functions and classes so that users have a single import point.

**Acceptance Criteria:**
- [ ] `orbiter/observability/__init__.py` exports all public symbols with `__all__`
- [ ] Tiered imports: core symbols (get_logger, configure, traced, span) importable from `orbiter.observability` directly
- [ ] Sub-module imports work: `from orbiter.observability.metrics import MetricsCollector`
- [ ] Lazy imports for OTel-dependent code (no ImportError if opentelemetry not installed)
- [ ] Convenience: `from orbiter.observability import get_logger, configure, traced` covers 90% of use cases
- [ ] Tests: import smoke tests with and without OTel, __all__ completeness check
- [ ] Pyright passes

## Functional Requirements

- FR-1: `orbiter-observability` has zero required external dependencies beyond `pydantic>=2.0`
- FR-2: `opentelemetry-api` and `opentelemetry-sdk` available as optional extras (`pip install orbiter-observability[otel]`)
- FR-3: All tracing/metrics code gracefully degrades to no-ops or in-memory collection when OTel is not installed
- FR-4: `configure()` is the single entry point for all observability setup — idempotent by default
- FR-5: `get_logger(name)` auto-prefixes names under `orbiter.*` namespace
- FR-6: `@traced` decorator works on sync/async functions and sync/async generators
- FR-7: `MetricsCollector` provides in-memory metric storage accessible via `get_metrics_snapshot()`
- FR-8: `HealthRegistry` aggregates multiple health checks into a single status
- FR-9: `AlertManager` enforces cooldown periods to prevent alert storms
- FR-10: `CostTracker` is thread-safe and supports custom model pricing registration
- FR-11: `SLOTracker` uses sliding time windows for compliance calculation
- FR-12: W3C Baggage propagation follows RFC 9110 with proper size limits
- FR-13: All modules use `contextvars` for async-safe state management
- FR-14: Package follows namespace package pattern (extends `orbiter.*` via hatchling wheel config)

## Non-Goals

- No auto-instrumentation or monkey-patching of orbiter packages
- No dashboard UI or visualization (that's orbiter-server's concern)
- No persistent storage of metrics/traces (use external collectors)
- No log file rotation or file-based log handlers (users can add their own)
- No Prometheus client library integration (OTel covers Prometheus export via collectors)
- No async log handlers (stdlib logging is synchronous; structured output is enough)
- No backwards compatibility with AWorld's `aworld.trace` or `aworld.logs` APIs

## Technical Considerations

- **Namespace packages:** Must use `src/orbiter/observability/` layout with hatchling `packages = ["src/orbiter"]` to extend the `orbiter.*` namespace
- **Optional imports:** Use try/except pattern for OpenTelemetry imports — define `HAS_OTEL` flag checked at runtime
- **Test isolation:** Tests for OTel-dependent code should use `pytest.importorskip("opentelemetry")` or mock the import
- **Test file naming:** Use `test_obs_*.py` prefix to avoid collision with orbiter-core's `test_log.py` etc.
- **Existing code to port:** ~900 lines from orbiter-trace (decorator, instrumentation, config, prompt_logger, propagation) + 87 lines from orbiter.log — total ~1000 lines to migrate, ~1000 lines new (health, alerts, cost, SLO)
- **Import path migration:** `orbiter.trace.*` → `orbiter.observability.*`, `orbiter.log` → `orbiter.observability.logging`
- **Backward compatibility shim:** `orbiter.log` in orbiter-core becomes a re-export module during transition

## Success Metrics

- All 887+ existing tests continue to pass after migration
- Zero external dependencies in base install (only pydantic)
- `from orbiter.observability import get_logger, configure, traced` works without OTel installed
- All 14 user stories have passing acceptance criteria
- Package adds < 2500 total lines of production code

## Open Questions

- Should the backward-compat shim in `orbiter.log` emit a deprecation warning, or silently re-export?
- Should cost pricing be loaded from a YAML/JSON file instead of hardcoded, to make updates easier?
- Should `SLOTracker` persist window data to disk for crash recovery, or is in-memory sufficient?
