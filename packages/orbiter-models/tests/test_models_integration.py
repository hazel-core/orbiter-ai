"""Integration tests for the orbiter.models public API.

All imports use the public surface (``orbiter.models``), not internal modules.
These tests validate that the package __init__.py re-exports are correct,
auto-registration works, and get_provider() produces functional providers.
"""

from __future__ import annotations

import pytest

import orbiter.models as models  # pyright: ignore[reportMissingImports]
import orbiter.models.types as model_types  # pyright: ignore[reportMissingImports]
from orbiter.models import (  # pyright: ignore[reportMissingImports]
    AnthropicProvider,
    ModelError,
    ModelProvider,
    OpenAIProvider,
    get_provider,
    model_registry,
)

EXPECTED_ALL = [
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

INTERNAL_HELPERS = [
    "_map_finish_reason",
    "_to_openai_messages",
    "_parse_response",
    "_parse_stream_chunk",
    "_map_stop_reason",
    "_build_messages",
    "_convert_tools",
]


class TestPublicAPI:
    """Verify the __all__ exports and re-export identity."""

    def test_all_exports_defined(self) -> None:
        assert sorted(models.__all__) == EXPECTED_ALL

    def test_all_exports_importable(self) -> None:
        for name in models.__all__:
            assert hasattr(models, name), f"{name!r} not importable from orbiter.models"

    def test_no_private_leakage(self) -> None:
        for name in INTERNAL_HELPERS:
            assert not hasattr(models, name), f"internal {name!r} leaked onto orbiter.models"

    def test_type_identity(self) -> None:
        assert models.ModelResponse is model_types.ModelResponse
        assert models.ModelError is model_types.ModelError
        assert models.StreamChunk is model_types.StreamChunk
        assert models.ToolCallDelta is model_types.ToolCallDelta


class TestAutoRegistration:
    """Verify providers are auto-registered on import."""

    def test_openai_registered(self) -> None:
        assert "openai" in model_registry
        assert model_registry.get("openai") is OpenAIProvider

    def test_anthropic_registered(self) -> None:
        assert "anthropic" in model_registry
        assert model_registry.get("anthropic") is AnthropicProvider

    def test_registry_lists_both(self) -> None:
        all_providers = model_registry.list_all()
        assert "openai" in all_providers
        assert "anthropic" in all_providers


class TestGetProviderEndToEnd:
    """End-to-end tests for get_provider() with real providers."""

    def test_openai_model_string(self) -> None:
        provider = get_provider("openai:gpt-4o", api_key="sk-test")
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.provider == "openai"
        assert provider.config.model_name == "gpt-4o"
        assert provider.config.api_key == "sk-test"

    def test_anthropic_model_string(self) -> None:
        provider = get_provider("anthropic:claude-sonnet-4-5-20250929", api_key="sk-ant")
        assert isinstance(provider, AnthropicProvider)
        assert provider.config.provider == "anthropic"
        assert provider.config.model_name == "claude-sonnet-4-5-20250929"
        assert provider.config.api_key == "sk-ant"

    def test_default_provider_is_openai(self) -> None:
        provider = get_provider("gpt-4o-mini", api_key="sk-test")
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.provider == "openai"
        assert provider.config.model_name == "gpt-4o-mini"

    def test_unknown_provider_error(self) -> None:
        with pytest.raises(ModelError, match="Available:"):
            get_provider("unknown:model")


class TestCrossProviderConsistency:
    """Verify both providers satisfy the same contract."""

    def test_both_are_model_providers(self) -> None:
        openai = get_provider("openai:gpt-4o", api_key="sk-test")
        anthropic = get_provider("anthropic:claude-sonnet-4-5-20250929", api_key="sk-ant")
        assert isinstance(openai, ModelProvider)
        assert isinstance(anthropic, ModelProvider)

    def test_both_have_required_methods(self) -> None:
        openai = get_provider("openai:gpt-4o", api_key="sk-test")
        anthropic = get_provider("anthropic:claude-sonnet-4-5-20250929", api_key="sk-ant")
        for provider in (openai, anthropic):
            assert callable(getattr(provider, "complete", None))
            assert callable(getattr(provider, "stream", None))
