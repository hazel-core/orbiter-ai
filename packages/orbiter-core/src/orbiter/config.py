"""Configuration types for the Orbiter framework."""

from __future__ import annotations

from pydantic import BaseModel, Field


def parse_model_string(model: str) -> tuple[str, str]:
    """Split a model string into provider and model name.

    Parses the ``"provider:model_name"`` format. If no colon is present,
    defaults the provider to ``"openai"``.

    Args:
        model: Model string, e.g. ``"openai:gpt-4o"`` or ``"gpt-4o"``.

    Returns:
        A ``(provider, model_name)`` tuple.
    """
    if ":" in model:
        provider, _, model_name = model.partition(":")
        return provider, model_name
    return "openai", model


class ModelConfig(BaseModel):
    """Configuration for an LLM provider connection.

    Args:
        provider: Provider name, e.g. ``"openai"`` or ``"anthropic"``.
        model_name: Model identifier within the provider.
        api_key: API key for authentication.
        base_url: Custom API base URL.
        max_retries: Maximum number of retries on transient failures.
        timeout: Request timeout in seconds.
    """

    model_config = {"frozen": True}

    provider: str = "openai"
    model_name: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    max_retries: int = Field(default=3, ge=0)
    timeout: float = Field(default=30.0, gt=0)


class AgentConfig(BaseModel):
    """Configuration for an Agent.

    Args:
        name: Unique identifier for the agent.
        model: Model string in ``"provider:model_name"`` format.
        instructions: System prompt for the agent.
        temperature: LLM sampling temperature.
        max_tokens: Maximum output tokens per LLM call.
        max_steps: Maximum LLM-tool round-trips.
    """

    model_config = {"frozen": True}

    name: str
    model: str = "openai:gpt-4o"
    instructions: str = ""
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = None
    max_steps: int = Field(default=10, ge=1)


class TaskConfig(BaseModel):
    """Configuration for a task.

    Args:
        name: Unique identifier for the task.
        description: Human-readable description of what the task does.
    """

    model_config = {"frozen": True}

    name: str
    description: str = ""


class RunConfig(BaseModel):
    """Configuration for a single run invocation.

    Args:
        max_steps: Maximum LLM-tool round-trips for this run.
        timeout: Overall timeout in seconds for the run.
        stream: Whether to enable streaming output.
        verbose: Whether to enable verbose logging.
    """

    model_config = {"frozen": True}

    max_steps: int = Field(default=10, ge=1)
    timeout: float | None = None
    stream: bool = False
    verbose: bool = False
