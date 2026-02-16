"""Orbiter Observability: structured logging, tracing, metrics, cost tracking.

Convenience imports -- the most common symbols are available directly::

    from orbiter.observability import get_logger, configure, traced, span

All other symbols are loaded lazily on first access so that importing this
package does not pull in OTel or other heavy sub-modules until needed.
Sub-module imports always work::

    from orbiter.observability.metrics import MetricsCollector
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Eager imports: core symbols that cover 90 % of use-cases
# ---------------------------------------------------------------------------
from orbiter.observability.config import (  # pyright: ignore[reportMissingImports]
    ObservabilityConfig,
    TraceBackend,
    configure,
    get_config,
)
from orbiter.observability.logging import (  # pyright: ignore[reportMissingImports]
    LogContext,
    configure_logging,
    get_logger,
)
from orbiter.observability.tracing import (  # pyright: ignore[reportMissingImports]
    aspan,
    span,
    traced,
)

# ---------------------------------------------------------------------------
# Lazy-loaded module mapping: attribute name -> (module_path, symbol_name)
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # config (reset/get_config already eager above)
    "reset": ("orbiter.observability.config", "reset"),
    # logging extras
    "TextFormatter": ("orbiter.observability.logging", "TextFormatter"),
    "JsonFormatter": ("orbiter.observability.logging", "JsonFormatter"),
    "reset_logging": ("orbiter.observability.logging", "reset_logging"),
    # tracing extras
    "SpanLike": ("orbiter.observability.tracing", "SpanLike"),
    "NullSpan": ("orbiter.observability.tracing", "NullSpan"),
    "is_user_code": ("orbiter.observability.tracing", "is_user_code"),
    "get_user_frame": ("orbiter.observability.tracing", "get_user_frame"),
    "extract_metadata": ("orbiter.observability.tracing", "extract_metadata"),
    # metrics
    "MetricsCollector": ("orbiter.observability.metrics", "MetricsCollector"),
    "METRIC_AGENT_RUN_DURATION": (
        "orbiter.observability.metrics",
        "METRIC_AGENT_RUN_DURATION",
    ),
    "METRIC_AGENT_RUN_COUNTER": (
        "orbiter.observability.metrics",
        "METRIC_AGENT_RUN_COUNTER",
    ),
    "METRIC_AGENT_TOKEN_USAGE": (
        "orbiter.observability.metrics",
        "METRIC_AGENT_TOKEN_USAGE",
    ),
    "METRIC_TOOL_STEP_DURATION": (
        "orbiter.observability.metrics",
        "METRIC_TOOL_STEP_DURATION",
    ),
    "METRIC_TOOL_STEP_COUNTER": (
        "orbiter.observability.metrics",
        "METRIC_TOOL_STEP_COUNTER",
    ),
    "get_collector": ("orbiter.observability.metrics", "get_collector"),
    "get_metrics_snapshot": (
        "orbiter.observability.metrics",
        "get_metrics_snapshot",
    ),
    "reset_metrics": ("orbiter.observability.metrics", "reset_metrics"),
    "create_counter": ("orbiter.observability.metrics", "create_counter"),
    "create_histogram": ("orbiter.observability.metrics", "create_histogram"),
    "create_gauge": ("orbiter.observability.metrics", "create_gauge"),
    "build_agent_attributes": (
        "orbiter.observability.metrics",
        "build_agent_attributes",
    ),
    "build_tool_attributes": (
        "orbiter.observability.metrics",
        "build_tool_attributes",
    ),
    "record_agent_run": ("orbiter.observability.metrics", "record_agent_run"),
    "record_tool_step": ("orbiter.observability.metrics", "record_tool_step"),
    "Timer": ("orbiter.observability.metrics", "Timer"),
    "timer": ("orbiter.observability.metrics", "timer"),
    # prompt logger
    "DEFAULT_CHAR_TOKEN_RATIO": (
        "orbiter.observability.prompt_logger",
        "DEFAULT_CHAR_TOKEN_RATIO",
    ),
    "estimate_tokens": (
        "orbiter.observability.prompt_logger",
        "estimate_tokens",
    ),
    "TokenBreakdown": (
        "orbiter.observability.prompt_logger",
        "TokenBreakdown",
    ),
    "compute_token_breakdown": (
        "orbiter.observability.prompt_logger",
        "compute_token_breakdown",
    ),
    "ExecutionLogEntry": (
        "orbiter.observability.prompt_logger",
        "ExecutionLogEntry",
    ),
    "PromptLogger": ("orbiter.observability.prompt_logger", "PromptLogger"),
    # propagation
    "BAGGAGE_HEADER": (
        "orbiter.observability.propagation",
        "BAGGAGE_HEADER",
    ),
    "MAX_HEADER_LENGTH": (
        "orbiter.observability.propagation",
        "MAX_HEADER_LENGTH",
    ),
    "MAX_PAIR_LENGTH": (
        "orbiter.observability.propagation",
        "MAX_PAIR_LENGTH",
    ),
    "MAX_PAIRS": ("orbiter.observability.propagation", "MAX_PAIRS"),
    "Carrier": ("orbiter.observability.propagation", "Carrier"),
    "DictCarrier": ("orbiter.observability.propagation", "DictCarrier"),
    "get_baggage": ("orbiter.observability.propagation", "get_baggage"),
    "get_baggage_value": (
        "orbiter.observability.propagation",
        "get_baggage_value",
    ),
    "set_baggage": ("orbiter.observability.propagation", "set_baggage"),
    "clear_baggage": ("orbiter.observability.propagation", "clear_baggage"),
    "BaggagePropagator": (
        "orbiter.observability.propagation",
        "BaggagePropagator",
    ),
    "SpanConsumer": ("orbiter.observability.propagation", "SpanConsumer"),
    "register_span_consumer": (
        "orbiter.observability.propagation",
        "register_span_consumer",
    ),
    "get_span_consumer": (
        "orbiter.observability.propagation",
        "get_span_consumer",
    ),
    "list_span_consumers": (
        "orbiter.observability.propagation",
        "list_span_consumers",
    ),
    "dispatch_spans": ("orbiter.observability.propagation", "dispatch_spans"),
    "clear_span_consumers": (
        "orbiter.observability.propagation",
        "clear_span_consumers",
    ),
    # health
    "HealthStatus": ("orbiter.observability.health", "HealthStatus"),
    "HealthResult": ("orbiter.observability.health", "HealthResult"),
    "HealthCheck": ("orbiter.observability.health", "HealthCheck"),
    "MemoryUsageCheck": ("orbiter.observability.health", "MemoryUsageCheck"),
    "EventLoopCheck": ("orbiter.observability.health", "EventLoopCheck"),
    "HealthRegistry": ("orbiter.observability.health", "HealthRegistry"),
    "get_registry": ("orbiter.observability.health", "get_registry"),
    "get_health_summary": (
        "orbiter.observability.health",
        "get_health_summary",
    ),
    # alerts
    "AlertSeverity": ("orbiter.observability.alerts", "AlertSeverity"),
    "Comparator": ("orbiter.observability.alerts", "Comparator"),
    "AlertRule": ("orbiter.observability.alerts", "AlertRule"),
    "Alert": ("orbiter.observability.alerts", "Alert"),
    "AlertCallback": ("orbiter.observability.alerts", "AlertCallback"),
    "AlertManager": ("orbiter.observability.alerts", "AlertManager"),
    "get_manager": ("orbiter.observability.alerts", "get_manager"),
    # cost
    "ModelPricing": ("orbiter.observability.cost", "ModelPricing"),
    "CostEntry": ("orbiter.observability.cost", "CostEntry"),
    "CostTracker": ("orbiter.observability.cost", "CostTracker"),
    "get_tracker": ("orbiter.observability.cost", "get_tracker"),
    # slo
    "SLO": ("orbiter.observability.slo", "SLO"),
    "SLOReport": ("orbiter.observability.slo", "SLOReport"),
    "SLOTracker": ("orbiter.observability.slo", "SLOTracker"),
}

# Semconv constants — all lazy-loaded from semconv module
_SEMCONV_NAMES: list[str] = [
    "GEN_AI_SYSTEM",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_REQUEST_MAX_TOKENS",
    "GEN_AI_REQUEST_TEMPERATURE",
    "GEN_AI_REQUEST_TOP_P",
    "GEN_AI_REQUEST_TOP_K",
    "GEN_AI_REQUEST_FREQUENCY_PENALTY",
    "GEN_AI_REQUEST_PRESENCE_PENALTY",
    "GEN_AI_REQUEST_STOP_SEQUENCES",
    "GEN_AI_REQUEST_STREAMING",
    "GEN_AI_PROMPT",
    "GEN_AI_COMPLETION",
    "GEN_AI_DURATION",
    "GEN_AI_RESPONSE_FINISH_REASONS",
    "GEN_AI_RESPONSE_ID",
    "GEN_AI_RESPONSE_MODEL",
    "GEN_AI_USAGE_INPUT_TOKENS",
    "GEN_AI_USAGE_OUTPUT_TOKENS",
    "GEN_AI_USAGE_TOTAL_TOKENS",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_SERVER_ADDRESS",
    "AGENT_ID",
    "AGENT_NAME",
    "AGENT_TYPE",
    "AGENT_MODEL",
    "AGENT_STEP",
    "AGENT_MAX_STEPS",
    "AGENT_RUN_SUCCESS",
    "TOOL_NAME",
    "TOOL_CALL_ID",
    "TOOL_ARGUMENTS",
    "TOOL_RESULT",
    "TOOL_ERROR",
    "TOOL_DURATION",
    "TOOL_STEP_SUCCESS",
    "TASK_ID",
    "TASK_INPUT",
    "SESSION_ID",
    "USER_ID",
    "TRACE_ID",
    "COST_INPUT_TOKENS",
    "COST_OUTPUT_TOKENS",
    "COST_TOTAL_USD",
    "SPAN_PREFIX_AGENT",
    "SPAN_PREFIX_TOOL",
    "SPAN_PREFIX_LLM",
    "SPAN_PREFIX_TASK",
]
for _name in _SEMCONV_NAMES:
    _LAZY_IMPORTS[_name] = ("orbiter.observability.semconv", _name)

# ---------------------------------------------------------------------------
# __all__ — everything available from this package
# ---------------------------------------------------------------------------
__all__: list[str] = [
    # config (eager)
    "ObservabilityConfig",
    "TraceBackend",
    "configure",
    "get_config",
    # logging (eager)
    "get_logger",
    "configure_logging",
    "LogContext",
    # tracing (eager)
    "traced",
    "span",
    "aspan",
    # lazy symbols
    *_LAZY_IMPORTS,
]


def __getattr__(name: str) -> object:
    """Lazy-load symbols on first access."""
    if name in _LAZY_IMPORTS:
        module_path, symbol_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, symbol_name)
        # Cache on this module so __getattr__ is only called once per symbol
        globals()[name] = value
        return value
    msg = f"module 'orbiter.observability' has no attribute {name!r}"
    raise AttributeError(msg)
