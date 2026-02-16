"""Agent and tool metrics instrumentation using OpenTelemetry."""

from __future__ import annotations

import time
from typing import Any

from opentelemetry import metrics

from orbiter.trace.config import (  # pyright: ignore[reportMissingImports]
    AGENT_NAME,
    AGENT_RUN_SUCCESS,
    AGENT_STEP,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    SESSION_ID,
    TASK_ID,
    TOOL_NAME,
    TOOL_STEP_SUCCESS,
    USER_ID,
)

# ---------------------------------------------------------------------------
# Metric names (constants for consistent instrument naming)
# ---------------------------------------------------------------------------

METRIC_AGENT_RUN_DURATION = "agent_run_duration"
METRIC_AGENT_RUN_COUNTER = "agent_run_counter"
METRIC_AGENT_TOKEN_USAGE = "agent_token_usage"
METRIC_TOOL_STEP_DURATION = "tool_step_duration"
METRIC_TOOL_STEP_COUNTER = "tool_step_counter"


# ---------------------------------------------------------------------------
# Instrument factory — creates instruments from the current meter provider
# ---------------------------------------------------------------------------


def _get_meter() -> metrics.Meter:
    """Get the orbiter meter from the current global MeterProvider."""
    return metrics.get_meter("orbiter")


def create_agent_run_duration() -> metrics.Histogram:
    """Create the agent_run_duration histogram."""
    return _get_meter().create_histogram(
        name=METRIC_AGENT_RUN_DURATION,
        unit="s",
        description="Agent run duration in seconds",
    )


def create_agent_run_counter() -> metrics.Counter:
    """Create the agent_run_counter counter."""
    return _get_meter().create_counter(
        name=METRIC_AGENT_RUN_COUNTER,
        unit="1",
        description="Number of agent run invocations",
    )


def create_agent_token_usage() -> metrics.Histogram:
    """Create the agent_token_usage histogram."""
    return _get_meter().create_histogram(
        name=METRIC_AGENT_TOKEN_USAGE,
        unit="token",
        description="Agent token usage per run",
    )


def create_tool_step_duration() -> metrics.Histogram:
    """Create the tool_step_duration histogram."""
    return _get_meter().create_histogram(
        name=METRIC_TOOL_STEP_DURATION,
        unit="s",
        description="Tool step execution duration in seconds",
    )


def create_tool_step_counter() -> metrics.Counter:
    """Create the tool_step_counter counter."""
    return _get_meter().create_counter(
        name=METRIC_TOOL_STEP_COUNTER,
        unit="1",
        description="Number of tool step invocations",
    )


# ---------------------------------------------------------------------------
# Attribute builders
# ---------------------------------------------------------------------------


def _safe_str(value: Any) -> str:
    """Convert a value to string, returning empty string for None."""
    return str(value) if value is not None else ""


def build_agent_attributes(
    *,
    agent_name: str,
    task_id: str = "",
    session_id: str = "",
    user_id: str = "",
    step: int | None = None,
) -> dict[str, str | int]:
    """Build attribute dict for agent metrics."""
    attrs: dict[str, str | int] = {
        AGENT_NAME: _safe_str(agent_name),
        TASK_ID: _safe_str(task_id),
        SESSION_ID: _safe_str(session_id),
        USER_ID: _safe_str(user_id),
    }
    if step is not None:
        attrs[AGENT_STEP] = step
    return attrs


def build_tool_attributes(
    *,
    tool_name: str,
    agent_name: str = "",
    task_id: str = "",
) -> dict[str, str]:
    """Build attribute dict for tool metrics."""
    return {
        TOOL_NAME: _safe_str(tool_name),
        AGENT_NAME: _safe_str(agent_name),
        TASK_ID: _safe_str(task_id),
    }


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------


def record_agent_run(
    *,
    duration: float,
    success: bool,
    attributes: dict[str, Any] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Record agent run metrics (duration, counter, token usage).

    Creates instruments from the current MeterProvider on each call so that
    metrics are recorded to whatever provider is active at call time.
    """
    attrs = dict(attributes) if attributes else {}
    attrs[AGENT_RUN_SUCCESS] = "1" if success else "0"
    meter = _get_meter()
    meter.create_histogram(
        name=METRIC_AGENT_RUN_DURATION,
        unit="s",
        description="Agent run duration in seconds",
    ).record(duration, attrs)
    meter.create_counter(
        name=METRIC_AGENT_RUN_COUNTER,
        unit="1",
        description="Number of agent run invocations",
    ).add(1, attrs)
    total_tokens = input_tokens + output_tokens
    if total_tokens > 0:
        token_attrs = dict(attrs)
        token_attrs[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
        token_attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
        token_attrs[GEN_AI_USAGE_TOTAL_TOKENS] = total_tokens
        meter.create_histogram(
            name=METRIC_AGENT_TOKEN_USAGE,
            unit="token",
            description="Agent token usage per run",
        ).record(total_tokens, token_attrs)


def record_tool_step(
    *,
    duration: float,
    success: bool,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record tool step metrics (duration, counter).

    Creates instruments from the current MeterProvider on each call so that
    metrics are recorded to whatever provider is active at call time.
    """
    attrs = dict(attributes) if attributes else {}
    attrs[TOOL_STEP_SUCCESS] = "1" if success else "0"
    meter = _get_meter()
    meter.create_histogram(
        name=METRIC_TOOL_STEP_DURATION,
        unit="s",
        description="Tool step execution duration in seconds",
    ).record(duration, attrs)
    meter.create_counter(
        name=METRIC_TOOL_STEP_COUNTER,
        unit="1",
        description="Number of tool step invocations",
    ).add(1, attrs)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


class Timer:
    """Simple timer for measuring durations.

    Usage::

        timer = Timer()
        timer.start()
        ...
        elapsed = timer.stop()
    """

    __slots__ = ("_elapsed", "_start")

    def __init__(self) -> None:
        self._start: float = 0.0
        self._elapsed: float = 0.0

    def start(self) -> Timer:
        """Start the timer. Returns self for chaining."""
        self._start = time.monotonic()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        self._elapsed = time.monotonic() - self._start
        return self._elapsed

    @property
    def elapsed(self) -> float:
        """Return the last recorded elapsed time."""
        return self._elapsed
