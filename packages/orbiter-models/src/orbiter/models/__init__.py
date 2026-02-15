"""Orbiter Models: LLM provider abstractions."""

# Import providers to trigger auto-registration with model_registry.
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .provider import ModelProvider, get_provider, model_registry
from .types import (
    FinishReason,
    ModelError,
    ModelResponse,
    StreamChunk,
    ToolCallDelta,
)

__all__ = [
    "AnthropicProvider",
    "FinishReason",
    "ModelError",
    "ModelProvider",
    "ModelResponse",
    "OpenAIProvider",
    "StreamChunk",
    "ToolCallDelta",
    "get_provider",
    "model_registry",
]
