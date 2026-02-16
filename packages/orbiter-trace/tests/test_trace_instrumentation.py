"""Tests for agent/tool instrumentation metrics."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

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
from orbiter.trace.instrumentation import (  # pyright: ignore[reportMissingImports]
    METRIC_AGENT_RUN_COUNTER,
    METRIC_AGENT_RUN_DURATION,
    METRIC_AGENT_TOKEN_USAGE,
    METRIC_TOOL_STEP_COUNTER,
    METRIC_TOOL_STEP_DURATION,
    Timer,
    build_agent_attributes,
    build_tool_attributes,
    create_agent_run_counter,
    create_agent_run_duration,
    create_agent_token_usage,
    create_tool_step_counter,
    create_tool_step_duration,
    record_agent_run,
    record_tool_step,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_meter_provider() -> None:
    """Reset the global MeterProvider before each test so set_meter_provider works."""
    metrics._internal._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    metrics._internal._METER_PROVIDER = None  # type: ignore[attr-defined]


@pytest.fixture()
def metric_reader() -> InMemoryMetricReader:
    """Create an in-memory metric reader and set up a fresh MeterProvider."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    return reader


def _get_metric_data(reader: InMemoryMetricReader, name: str) -> list[dict[str, Any]]:
    """Extract data points for a named metric from the reader."""
    data = reader.get_metrics_data()
    results: list[dict[str, Any]] = []
    if data is None:
        return results
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name:
                    for dp in metric.data.data_points:
                        results.append(
                            {
                                "value": getattr(dp, "value", None) or getattr(dp, "sum", None),
                                "attributes": dict(dp.attributes) if dp.attributes else {},
                            }
                        )
    return results


# ---------------------------------------------------------------------------
# Metric name constant tests
# ---------------------------------------------------------------------------


class TestMetricNames:
    """Verify metric name constants."""

    def test_agent_run_duration_name(self) -> None:
        assert METRIC_AGENT_RUN_DURATION == "agent_run_duration"

    def test_agent_run_counter_name(self) -> None:
        assert METRIC_AGENT_RUN_COUNTER == "agent_run_counter"

    def test_agent_token_usage_name(self) -> None:
        assert METRIC_AGENT_TOKEN_USAGE == "agent_token_usage"

    def test_tool_step_duration_name(self) -> None:
        assert METRIC_TOOL_STEP_DURATION == "tool_step_duration"

    def test_tool_step_counter_name(self) -> None:
        assert METRIC_TOOL_STEP_COUNTER == "tool_step_counter"


# ---------------------------------------------------------------------------
# Instrument factory tests
# ---------------------------------------------------------------------------


class TestInstrumentFactory:
    """Verify instrument factory functions return valid instruments."""

    def test_agent_run_duration_creates_histogram(self) -> None:
        h = create_agent_run_duration()
        assert h is not None

    def test_agent_run_counter_creates_counter(self) -> None:
        c = create_agent_run_counter()
        assert c is not None

    def test_agent_token_usage_creates_histogram(self) -> None:
        h = create_agent_token_usage()
        assert h is not None

    def test_tool_step_duration_creates_histogram(self) -> None:
        h = create_tool_step_duration()
        assert h is not None

    def test_tool_step_counter_creates_counter(self) -> None:
        c = create_tool_step_counter()
        assert c is not None


# ---------------------------------------------------------------------------
# Attribute builder tests
# ---------------------------------------------------------------------------


class TestBuildAgentAttributes:
    """Test build_agent_attributes helper."""

    def test_defaults(self) -> None:
        attrs = build_agent_attributes(agent_name="test-agent")
        assert attrs[AGENT_NAME] == "test-agent"
        assert attrs[TASK_ID] == ""
        assert attrs[SESSION_ID] == ""
        assert attrs[USER_ID] == ""
        assert AGENT_STEP not in attrs

    def test_all_fields(self) -> None:
        attrs = build_agent_attributes(
            agent_name="my-agent",
            task_id="task-1",
            session_id="sess-1",
            user_id="user-1",
            step=3,
        )
        assert attrs[AGENT_NAME] == "my-agent"
        assert attrs[TASK_ID] == "task-1"
        assert attrs[SESSION_ID] == "sess-1"
        assert attrs[USER_ID] == "user-1"
        assert attrs[AGENT_STEP] == 3

    def test_none_values_become_empty_string(self) -> None:
        attrs = build_agent_attributes(agent_name="a", task_id=None)  # type: ignore[arg-type]
        assert attrs[TASK_ID] == ""


class TestBuildToolAttributes:
    """Test build_tool_attributes helper."""

    def test_defaults(self) -> None:
        attrs = build_tool_attributes(tool_name="search")
        assert attrs[TOOL_NAME] == "search"
        assert attrs[AGENT_NAME] == ""
        assert attrs[TASK_ID] == ""

    def test_all_fields(self) -> None:
        attrs = build_tool_attributes(
            tool_name="search",
            agent_name="my-agent",
            task_id="task-1",
        )
        assert attrs[TOOL_NAME] == "search"
        assert attrs[AGENT_NAME] == "my-agent"
        assert attrs[TASK_ID] == "task-1"


# ---------------------------------------------------------------------------
# Recording tests (with in-memory metric reader)
# ---------------------------------------------------------------------------


class TestRecordAgentRun:
    """Test record_agent_run records to all three agent metrics."""

    def test_success(self, metric_reader: InMemoryMetricReader) -> None:
        attrs = build_agent_attributes(agent_name="agent-a")
        record_agent_run(
            duration=1.5,
            success=True,
            attributes=attrs,
            input_tokens=100,
            output_tokens=50,
        )
        dur_data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_DURATION)
        assert len(dur_data) > 0

        cnt_data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_COUNTER)
        assert len(cnt_data) > 0

        tok_data = _get_metric_data(metric_reader, METRIC_AGENT_TOKEN_USAGE)
        assert len(tok_data) > 0

    def test_failure_sets_success_zero(self, metric_reader: InMemoryMetricReader) -> None:
        attrs = build_agent_attributes(agent_name="agent-b")
        record_agent_run(duration=0.3, success=False, attributes=attrs)
        dur_data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_DURATION)
        assert len(dur_data) > 0
        assert dur_data[0]["attributes"][AGENT_RUN_SUCCESS] == "0"

    def test_no_tokens_skips_token_metric(self, metric_reader: InMemoryMetricReader) -> None:
        record_agent_run(duration=0.1, success=True)
        tok_data = _get_metric_data(metric_reader, METRIC_AGENT_TOKEN_USAGE)
        assert len(tok_data) == 0

    def test_token_attributes(self, metric_reader: InMemoryMetricReader) -> None:
        record_agent_run(
            duration=1.0,
            success=True,
            input_tokens=200,
            output_tokens=100,
        )
        tok_data = _get_metric_data(metric_reader, METRIC_AGENT_TOKEN_USAGE)
        assert len(tok_data) > 0
        attrs = tok_data[0]["attributes"]
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 200
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 100
        assert attrs[GEN_AI_USAGE_TOTAL_TOKENS] == 300

    def test_empty_attributes(self, metric_reader: InMemoryMetricReader) -> None:
        record_agent_run(duration=0.5, success=True)
        dur_data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_DURATION)
        assert len(dur_data) > 0
        assert dur_data[0]["attributes"][AGENT_RUN_SUCCESS] == "1"


class TestRecordToolStep:
    """Test record_tool_step records to both tool metrics."""

    def test_success(self, metric_reader: InMemoryMetricReader) -> None:
        attrs = build_tool_attributes(tool_name="search", agent_name="agent-a")
        record_tool_step(duration=0.2, success=True, attributes=attrs)

        dur_data = _get_metric_data(metric_reader, METRIC_TOOL_STEP_DURATION)
        assert len(dur_data) > 0

        cnt_data = _get_metric_data(metric_reader, METRIC_TOOL_STEP_COUNTER)
        assert len(cnt_data) > 0

    def test_failure(self, metric_reader: InMemoryMetricReader) -> None:
        attrs = build_tool_attributes(tool_name="fetch")
        record_tool_step(duration=0.5, success=False, attributes=attrs)
        dur_data = _get_metric_data(metric_reader, METRIC_TOOL_STEP_DURATION)
        assert len(dur_data) > 0
        assert dur_data[0]["attributes"][TOOL_STEP_SUCCESS] == "0"

    def test_empty_attributes(self, metric_reader: InMemoryMetricReader) -> None:
        record_tool_step(duration=0.1, success=True)
        cnt_data = _get_metric_data(metric_reader, METRIC_TOOL_STEP_COUNTER)
        assert len(cnt_data) > 0
        assert cnt_data[0]["attributes"][TOOL_STEP_SUCCESS] == "1"

    def test_does_not_mutate_input(self, metric_reader: InMemoryMetricReader) -> None:
        attrs = build_tool_attributes(tool_name="search")
        original = dict(attrs)
        record_tool_step(duration=0.1, success=True, attributes=attrs)
        assert attrs == original


# ---------------------------------------------------------------------------
# Timer tests
# ---------------------------------------------------------------------------


class TestTimer:
    """Test the Timer helper."""

    def test_basic_timing(self) -> None:
        import time

        timer = Timer()
        timer.start()
        time.sleep(0.01)
        elapsed = timer.stop()
        assert elapsed >= 0.005

    def test_start_returns_self(self) -> None:
        timer = Timer()
        assert timer.start() is timer

    def test_elapsed_property(self) -> None:
        timer = Timer()
        assert timer.elapsed == 0.0
        timer.start()
        timer.stop()
        assert timer.elapsed > 0.0

    def test_elapsed_matches_stop_return(self) -> None:
        timer = Timer()
        timer.start()
        result = timer.stop()
        assert timer.elapsed == result

    def test_initial_state(self) -> None:
        timer = Timer()
        assert timer.elapsed == 0.0


# ---------------------------------------------------------------------------
# Histogram bucket tests
# ---------------------------------------------------------------------------


class TestHistogramBuckets:
    """Test that histogram instruments record values within expected ranges."""

    def test_agent_duration_records_fractional(self, metric_reader: InMemoryMetricReader) -> None:
        record_agent_run(duration=0.123, success=True)
        data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_DURATION)
        assert len(data) > 0

    def test_agent_duration_records_large(self, metric_reader: InMemoryMetricReader) -> None:
        record_agent_run(duration=60.0, success=True)
        data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_DURATION)
        assert len(data) > 0

    def test_tool_duration_records_small(self, metric_reader: InMemoryMetricReader) -> None:
        record_tool_step(duration=0.001, success=True)
        data = _get_metric_data(metric_reader, METRIC_TOOL_STEP_DURATION)
        assert len(data) > 0

    def test_token_usage_records_value(self, metric_reader: InMemoryMetricReader) -> None:
        record_agent_run(duration=1.0, success=True, input_tokens=500, output_tokens=200)
        data = _get_metric_data(metric_reader, METRIC_AGENT_TOKEN_USAGE)
        assert len(data) > 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining attribute builders + recording."""

    def test_agent_full_flow(self, metric_reader: InMemoryMetricReader) -> None:
        attrs = build_agent_attributes(
            agent_name="planner",
            task_id="t-1",
            session_id="s-1",
            step=2,
        )
        timer = Timer().start()
        elapsed = timer.stop()
        record_agent_run(
            duration=elapsed,
            success=True,
            attributes=attrs,
            input_tokens=1000,
            output_tokens=500,
        )
        dur_data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_DURATION)
        assert len(dur_data) > 0
        assert dur_data[0]["attributes"][AGENT_NAME] == "planner"

    def test_tool_full_flow(self, metric_reader: InMemoryMetricReader) -> None:
        attrs = build_tool_attributes(
            tool_name="web_search",
            agent_name="researcher",
            task_id="t-1",
        )
        timer = Timer().start()
        elapsed = timer.stop()
        record_tool_step(duration=elapsed, success=True, attributes=attrs)
        dur_data = _get_metric_data(metric_reader, METRIC_TOOL_STEP_DURATION)
        assert len(dur_data) > 0
        assert dur_data[0]["attributes"][TOOL_NAME] == "web_search"

    def test_multiple_recordings(self, metric_reader: InMemoryMetricReader) -> None:
        for i in range(3):
            record_agent_run(
                duration=float(i),
                success=True,
                attributes=build_agent_attributes(agent_name=f"agent-{i}"),
            )
        cnt_data = _get_metric_data(metric_reader, METRIC_AGENT_RUN_COUNTER)
        assert len(cnt_data) > 0
