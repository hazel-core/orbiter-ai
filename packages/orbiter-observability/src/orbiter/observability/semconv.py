"""Semantic conventions for agent, tool, LLM, and task observability.

Standardised attribute names used across spans, metrics, and log records
so that all Orbiter telemetry is consistent and queryable.

Ported from ``orbiter.trace.config`` with new cost conventions added.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GenAI semantic conventions — gen_ai.*
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
# Agent conventions — orbiter.agent.*
# ---------------------------------------------------------------------------

AGENT_ID = "orbiter.agent.id"
AGENT_NAME = "orbiter.agent.name"
AGENT_TYPE = "orbiter.agent.type"
AGENT_MODEL = "orbiter.agent.model"
AGENT_STEP = "orbiter.agent.step"
AGENT_MAX_STEPS = "orbiter.agent.max_steps"
AGENT_RUN_SUCCESS = "orbiter.agent.run.success"

# ---------------------------------------------------------------------------
# Tool conventions — orbiter.tool.*
# ---------------------------------------------------------------------------

TOOL_NAME = "orbiter.tool.name"
TOOL_CALL_ID = "orbiter.tool.call_id"
TOOL_ARGUMENTS = "orbiter.tool.arguments"
TOOL_RESULT = "orbiter.tool.result"
TOOL_ERROR = "orbiter.tool.error"
TOOL_DURATION = "orbiter.tool.duration"
TOOL_STEP_SUCCESS = "orbiter.tool.step.success"

# ---------------------------------------------------------------------------
# Task / session / user conventions
# ---------------------------------------------------------------------------

TASK_ID = "orbiter.task.id"
TASK_INPUT = "orbiter.task.input"
SESSION_ID = "orbiter.session.id"
USER_ID = "orbiter.user.id"
TRACE_ID = "orbiter.trace.id"

# ---------------------------------------------------------------------------
# Cost conventions (new) — orbiter.cost.*
# ---------------------------------------------------------------------------

COST_INPUT_TOKENS = "orbiter.cost.input_tokens"
COST_OUTPUT_TOKENS = "orbiter.cost.output_tokens"
COST_TOTAL_USD = "orbiter.cost.total_usd"

# ---------------------------------------------------------------------------
# Distributed task conventions — orbiter.distributed.*
# ---------------------------------------------------------------------------

DIST_TASK_ID = "orbiter.distributed.task_id"
DIST_WORKER_ID = "orbiter.distributed.worker_id"
DIST_QUEUE_NAME = "orbiter.distributed.queue_name"
DIST_TASK_STATUS = "orbiter.distributed.task_status"

# ---------------------------------------------------------------------------
# Distributed task metric names
# ---------------------------------------------------------------------------

METRIC_DIST_TASKS_SUBMITTED = "dist_tasks_submitted"
METRIC_DIST_TASKS_COMPLETED = "dist_tasks_completed"
METRIC_DIST_TASKS_FAILED = "dist_tasks_failed"
METRIC_DIST_TASKS_CANCELLED = "dist_tasks_cancelled"
METRIC_DIST_QUEUE_DEPTH = "dist_queue_depth"
METRIC_DIST_TASK_DURATION = "dist_task_duration"
METRIC_DIST_TASK_WAIT_TIME = "dist_task_wait_time"

# ---------------------------------------------------------------------------
# Span name prefixes
# ---------------------------------------------------------------------------

SPAN_PREFIX_AGENT = "agent."
SPAN_PREFIX_TOOL = "tool."
SPAN_PREFIX_LLM = "llm."
SPAN_PREFIX_TASK = "task."
