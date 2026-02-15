"""Orbiter Trace: OpenTelemetry-based observability."""

from orbiter.trace.config import (  # pyright: ignore[reportMissingImports]
    TraceBackend,
    TraceConfig,
)
from orbiter.trace.decorator import (  # pyright: ignore[reportMissingImports]
    span_sync,
    traced,
)
from orbiter.trace.instrumentation import (  # pyright: ignore[reportMissingImports]
    Timer,
    build_agent_attributes,
    build_tool_attributes,
    record_agent_run,
    record_tool_step,
)
from orbiter.trace.prompt_logger import (  # pyright: ignore[reportMissingImports]
    ExecutionLogEntry,
    PromptLogger,
    TokenBreakdown,
    compute_token_breakdown,
    estimate_tokens,
)
from orbiter.trace.propagation import (  # pyright: ignore[reportMissingImports]
    BaggagePropagator,
    Carrier,
    DictCarrier,
    SpanConsumer,
    clear_baggage,
    clear_span_consumers,
    dispatch_spans,
    get_baggage,
    get_baggage_value,
    get_span_consumer,
    list_span_consumers,
    register_span_consumer,
    set_baggage,
)

__all__ = [
    "BaggagePropagator",
    "Carrier",
    "DictCarrier",
    "ExecutionLogEntry",
    "PromptLogger",
    "SpanConsumer",
    "Timer",
    "TokenBreakdown",
    "TraceBackend",
    "TraceConfig",
    "build_agent_attributes",
    "build_tool_attributes",
    "clear_baggage",
    "clear_span_consumers",
    "compute_token_breakdown",
    "dispatch_spans",
    "estimate_tokens",
    "get_baggage",
    "get_baggage_value",
    "get_span_consumer",
    "list_span_consumers",
    "record_agent_run",
    "record_tool_step",
    "register_span_consumer",
    "set_baggage",
    "span_sync",
    "traced",
]
