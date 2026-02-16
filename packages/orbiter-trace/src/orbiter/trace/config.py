"""Trace configuration and semantic conventions for agent/tool observability."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Trace backend selection
# ---------------------------------------------------------------------------


class TraceBackend(StrEnum):
    """Supported trace export backends."""

    OTLP = "otlp"
    MEMORY = "memory"
    CONSOLE = "console"


# ---------------------------------------------------------------------------
# Trace configuration
# ---------------------------------------------------------------------------


class TraceConfig(BaseModel, frozen=True):
    """Immutable configuration for the trace / observability layer.

    Controls backend selection, sampling, export endpoint, and attribute
    namespace used for all Orbiter-specific span attributes.
    """

    backend: TraceBackend = Field(
        default=TraceBackend.OTLP,
        description="Trace export backend",
    )

    endpoint: str | None = Field(
        default=None,
        description="OTLP collector endpoint (e.g. http://localhost:4318)",
    )

    service_name: str = Field(
        default="orbiter",
        description="Service name reported in exported spans",
    )

    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of sampling a trace (0.0 = none, 1.0 = all)",
    )

    enabled: bool = Field(
        default=True,
        description="Global toggle — when False, tracing is a no-op",
    )

    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Extra headers sent with each export request",
    )

    namespace: str = Field(
        default="orbiter",
        description="Attribute namespace prefix (e.g. orbiter.agent.name)",
    )

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Extension point for custom exporter/processor config",
    )

    @model_validator(mode="after")
    def _validate_endpoint(self) -> TraceConfig:
        """OTLP backend should have an endpoint (warn-only: no hard error)."""
        return self


# ---------------------------------------------------------------------------
# Semantic conventions — gen_ai.*
# ---------------------------------------------------------------------------

# Standard OpenTelemetry GenAI semantic conventions.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
GEN_AI_REQUEST_STREAMING = "gen_ai.request.streaming"
GEN_AI_PROMPT = "gen_ai.prompt"
GEN_AI_COMPLETION = "gen_ai.completion"
GEN_AI_DURATION = "gen_ai.duration"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_SERVER_ADDRESS = "gen_ai.server.address"

# ---------------------------------------------------------------------------
# Semantic conventions — agent.*
# ---------------------------------------------------------------------------

AGENT_ID = "orbiter.agent.id"
AGENT_NAME = "orbiter.agent.name"
AGENT_TYPE = "orbiter.agent.type"
AGENT_MODEL = "orbiter.agent.model"
AGENT_STEP = "orbiter.agent.step"
AGENT_MAX_STEPS = "orbiter.agent.max_steps"
AGENT_RUN_SUCCESS = "orbiter.agent.run.success"

# ---------------------------------------------------------------------------
# Semantic conventions — tool.*
# ---------------------------------------------------------------------------

TOOL_NAME = "orbiter.tool.name"
TOOL_CALL_ID = "orbiter.tool.call_id"
TOOL_ARGUMENTS = "orbiter.tool.arguments"
TOOL_RESULT = "orbiter.tool.result"
TOOL_ERROR = "orbiter.tool.error"
TOOL_DURATION = "orbiter.tool.duration"
TOOL_STEP_SUCCESS = "orbiter.tool.step.success"

# ---------------------------------------------------------------------------
# Semantic conventions — task / session / user
# ---------------------------------------------------------------------------

TASK_ID = "orbiter.task.id"
TASK_INPUT = "orbiter.task.input"
SESSION_ID = "orbiter.session.id"
USER_ID = "orbiter.user.id"
TRACE_ID = "orbiter.trace.id"

# ---------------------------------------------------------------------------
# Span name prefixes
# ---------------------------------------------------------------------------

SPAN_PREFIX_AGENT = "agent."
SPAN_PREFIX_TOOL = "tool."
SPAN_PREFIX_LLM = "llm."
SPAN_PREFIX_TASK = "task."
