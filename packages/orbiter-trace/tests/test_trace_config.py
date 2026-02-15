"""Tests for orbiter.trace.config — TraceConfig and semantic conventions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orbiter.trace.config import (  # pyright: ignore[reportMissingImports]
    AGENT_ID,
    AGENT_MODEL,
    AGENT_NAME,
    AGENT_RUN_SUCCESS,
    AGENT_STEP,
    AGENT_TYPE,
    GEN_AI_COMPLETION,
    GEN_AI_DURATION,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROMPT,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_STREAMING,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SERVER_ADDRESS,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_TOTAL_TOKENS,
    SESSION_ID,
    SPAN_PREFIX_AGENT,
    SPAN_PREFIX_LLM,
    SPAN_PREFIX_TASK,
    SPAN_PREFIX_TOOL,
    TASK_ID,
    TASK_INPUT,
    TOOL_ARGUMENTS,
    TOOL_CALL_ID,
    TOOL_DURATION,
    TOOL_ERROR,
    TOOL_NAME,
    TOOL_RESULT,
    TOOL_STEP_SUCCESS,
    TRACE_ID,
    USER_ID,
    TraceBackend,
    TraceConfig,
)

# ---------------------------------------------------------------------------
# TraceBackend enum
# ---------------------------------------------------------------------------


class TestTraceBackend:
    def test_values(self) -> None:
        assert TraceBackend.OTLP == "otlp"
        assert TraceBackend.MEMORY == "memory"
        assert TraceBackend.CONSOLE == "console"

    def test_is_str_enum(self) -> None:
        assert isinstance(TraceBackend.OTLP, str)


# ---------------------------------------------------------------------------
# TraceConfig defaults
# ---------------------------------------------------------------------------


class TestTraceConfigDefaults:
    def test_default_values(self) -> None:
        cfg = TraceConfig()
        assert cfg.backend == TraceBackend.OTLP
        assert cfg.endpoint is None
        assert cfg.service_name == "orbiter"
        assert cfg.sample_rate == 1.0
        assert cfg.enabled is True
        assert cfg.headers == {}
        assert cfg.namespace == "orbiter"
        assert cfg.extra == {}

    def test_frozen(self) -> None:
        cfg = TraceConfig()
        with pytest.raises(ValidationError):
            cfg.backend = TraceBackend.MEMORY  # type: ignore[misc]

    def test_repr_contains_backend(self) -> None:
        cfg = TraceConfig()
        r = repr(cfg)
        assert "otlp" in r


# ---------------------------------------------------------------------------
# TraceConfig custom values
# ---------------------------------------------------------------------------


class TestTraceConfigCustom:
    def test_custom_backend(self) -> None:
        cfg = TraceConfig(backend=TraceBackend.MEMORY)
        assert cfg.backend == TraceBackend.MEMORY

    def test_custom_endpoint(self) -> None:
        cfg = TraceConfig(endpoint="http://localhost:4318")
        assert cfg.endpoint == "http://localhost:4318"

    def test_custom_service_name(self) -> None:
        cfg = TraceConfig(service_name="my-service")
        assert cfg.service_name == "my-service"

    def test_custom_sample_rate(self) -> None:
        cfg = TraceConfig(sample_rate=0.5)
        assert cfg.sample_rate == 0.5

    def test_disabled(self) -> None:
        cfg = TraceConfig(enabled=False)
        assert cfg.enabled is False

    def test_custom_headers(self) -> None:
        cfg = TraceConfig(headers={"Authorization": "Bearer tok"})
        assert cfg.headers == {"Authorization": "Bearer tok"}

    def test_custom_namespace(self) -> None:
        cfg = TraceConfig(namespace="myapp")
        assert cfg.namespace == "myapp"

    def test_extra(self) -> None:
        cfg = TraceConfig(extra={"batch_size": 512})
        assert cfg.extra["batch_size"] == 512

    def test_backend_from_string(self) -> None:
        cfg = TraceConfig(backend="console")  # type: ignore[arg-type]
        assert cfg.backend == TraceBackend.CONSOLE


# ---------------------------------------------------------------------------
# TraceConfig validation
# ---------------------------------------------------------------------------


class TestTraceConfigValidation:
    def test_sample_rate_too_low(self) -> None:
        with pytest.raises(ValidationError, match="sample_rate"):
            TraceConfig(sample_rate=-0.1)

    def test_sample_rate_too_high(self) -> None:
        with pytest.raises(ValidationError, match="sample_rate"):
            TraceConfig(sample_rate=1.5)

    def test_sample_rate_boundary_zero(self) -> None:
        cfg = TraceConfig(sample_rate=0.0)
        assert cfg.sample_rate == 0.0

    def test_sample_rate_boundary_one(self) -> None:
        cfg = TraceConfig(sample_rate=1.0)
        assert cfg.sample_rate == 1.0

    def test_invalid_backend(self) -> None:
        with pytest.raises(ValidationError):
            TraceConfig(backend="nonexistent")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TraceConfig serialisation
# ---------------------------------------------------------------------------


class TestTraceConfigSerialisation:
    def test_model_dump(self) -> None:
        cfg = TraceConfig(backend=TraceBackend.MEMORY, service_name="test")
        d = cfg.model_dump()
        assert d["backend"] == "memory"
        assert d["service_name"] == "test"
        assert d["sample_rate"] == 1.0

    def test_model_dump_json_roundtrip(self) -> None:
        cfg = TraceConfig(endpoint="http://localhost:4318", headers={"X": "Y"})
        json_str = cfg.model_dump_json()
        restored = TraceConfig.model_validate_json(json_str)
        assert restored == cfg


# ---------------------------------------------------------------------------
# Semantic conventions — gen_ai.*
# ---------------------------------------------------------------------------


class TestGenAIConventions:
    def test_system(self) -> None:
        assert GEN_AI_SYSTEM == "gen_ai.system"

    def test_request_model(self) -> None:
        assert GEN_AI_REQUEST_MODEL == "gen_ai.request.model"

    def test_request_max_tokens(self) -> None:
        assert GEN_AI_REQUEST_MAX_TOKENS == "gen_ai.request.max_tokens"

    def test_request_temperature(self) -> None:
        assert GEN_AI_REQUEST_TEMPERATURE == "gen_ai.request.temperature"

    def test_request_top_p(self) -> None:
        assert GEN_AI_REQUEST_TOP_P == "gen_ai.request.top_p"

    def test_request_streaming(self) -> None:
        assert GEN_AI_REQUEST_STREAMING == "gen_ai.request.streaming"

    def test_prompt_and_completion(self) -> None:
        assert GEN_AI_PROMPT == "gen_ai.prompt"
        assert GEN_AI_COMPLETION == "gen_ai.completion"

    def test_duration(self) -> None:
        assert GEN_AI_DURATION == "gen_ai.duration"

    def test_response_attrs(self) -> None:
        assert GEN_AI_RESPONSE_FINISH_REASONS == "gen_ai.response.finish_reasons"
        assert GEN_AI_RESPONSE_ID == "gen_ai.response.id"
        assert GEN_AI_RESPONSE_MODEL == "gen_ai.response.model"

    def test_usage_tokens(self) -> None:
        assert GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
        assert GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
        assert GEN_AI_USAGE_TOTAL_TOKENS == "gen_ai.usage.total_tokens"

    def test_operation_and_server(self) -> None:
        assert GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
        assert GEN_AI_SERVER_ADDRESS == "gen_ai.server.address"


# ---------------------------------------------------------------------------
# Semantic conventions — agent.*
# ---------------------------------------------------------------------------


class TestAgentConventions:
    def test_agent_id(self) -> None:
        assert AGENT_ID == "orbiter.agent.id"

    def test_agent_name(self) -> None:
        assert AGENT_NAME == "orbiter.agent.name"

    def test_agent_type(self) -> None:
        assert AGENT_TYPE == "orbiter.agent.type"

    def test_agent_model(self) -> None:
        assert AGENT_MODEL == "orbiter.agent.model"

    def test_agent_step(self) -> None:
        assert AGENT_STEP == "orbiter.agent.step"

    def test_agent_run_success(self) -> None:
        assert AGENT_RUN_SUCCESS == "orbiter.agent.run.success"


# ---------------------------------------------------------------------------
# Semantic conventions — tool.*
# ---------------------------------------------------------------------------


class TestToolConventions:
    def test_tool_name(self) -> None:
        assert TOOL_NAME == "orbiter.tool.name"

    def test_tool_call_id(self) -> None:
        assert TOOL_CALL_ID == "orbiter.tool.call_id"

    def test_tool_arguments(self) -> None:
        assert TOOL_ARGUMENTS == "orbiter.tool.arguments"

    def test_tool_result(self) -> None:
        assert TOOL_RESULT == "orbiter.tool.result"

    def test_tool_error(self) -> None:
        assert TOOL_ERROR == "orbiter.tool.error"

    def test_tool_duration(self) -> None:
        assert TOOL_DURATION == "orbiter.tool.duration"

    def test_tool_step_success(self) -> None:
        assert TOOL_STEP_SUCCESS == "orbiter.tool.step.success"


# ---------------------------------------------------------------------------
# Semantic conventions — task / session / user / trace
# ---------------------------------------------------------------------------


class TestTaskSessionConventions:
    def test_task_id(self) -> None:
        assert TASK_ID == "orbiter.task.id"

    def test_task_input(self) -> None:
        assert TASK_INPUT == "orbiter.task.input"

    def test_session_id(self) -> None:
        assert SESSION_ID == "orbiter.session.id"

    def test_user_id(self) -> None:
        assert USER_ID == "orbiter.user.id"

    def test_trace_id(self) -> None:
        assert TRACE_ID == "orbiter.trace.id"


# ---------------------------------------------------------------------------
# Span name prefixes
# ---------------------------------------------------------------------------


class TestSpanPrefixes:
    def test_agent_prefix(self) -> None:
        assert SPAN_PREFIX_AGENT == "agent."

    def test_tool_prefix(self) -> None:
        assert SPAN_PREFIX_TOOL == "tool."

    def test_llm_prefix(self) -> None:
        assert SPAN_PREFIX_LLM == "llm."

    def test_task_prefix(self) -> None:
        assert SPAN_PREFIX_TASK == "task."

    def test_prefix_usage(self) -> None:
        """Prefixes can be concatenated with names to form span names."""
        assert SPAN_PREFIX_AGENT + "my-agent" == "agent.my-agent"
        assert SPAN_PREFIX_TOOL + "search" == "tool.search"
        assert SPAN_PREFIX_LLM + "gpt-4o" == "llm.gpt-4o"
