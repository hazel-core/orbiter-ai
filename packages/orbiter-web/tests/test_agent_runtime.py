"""Tests for the agent runtime bridge service."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orbiter.models.types import StreamChunk  # pyright: ignore[reportMissingImports]
from orbiter.types import AgentOutput, Usage, UserMessage
from orbiter_web.services.agent_runtime import (
    AgentRuntimeError,
    AgentService,
    _load_agent_row,
    _resolve_provider,
    _resolve_tools,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_AGENT_ROW = {
    "id": "agent-001",
    "name": "Test Agent",
    "description": "A test agent",
    "instructions": "You are a helpful assistant.",
    "model_provider": "openai",
    "model_name": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1024,
    "max_steps": 5,
    "output_type_json": "{}",
    "tools_json": "[]",
    "handoffs_json": "[]",
    "hooks_json": "{}",
    "project_id": "proj-001",
    "user_id": "user-001",
    "created_at": "2025-01-01T00:00:00",
    "updated_at": "2025-01-01T00:00:00",
}

FAKE_PROVIDER_ROW = {
    "id": "prov-001",
    "name": "OpenAI",
    "provider_type": "openai",
    "encrypted_api_key": "encrypted-key-123",
    "base_url": None,
    "max_retries": 3,
    "timeout": 30,
    "user_id": "user-001",
    "created_at": "2025-01-01T00:00:00",
    "updated_at": "2025-01-01T00:00:00",
}

FAKE_TOOL_ROW = {
    "id": "tool-001",
    "name": "search_web",
    "description": "Search the web for information",
    "category": "search",
    "schema_json": json.dumps({
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }),
    "code": "",
    "tool_type": "function",
    "usage_count": 0,
    "project_id": "proj-001",
    "user_id": "user-001",
    "created_at": "2025-01-01T00:00:00",
}


class FakeRow:
    """A mock aiosqlite.Row that supports dict() conversion and key access."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> Any:
        return self._data.keys()

    def __iter__(self) -> Any:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _make_mock_db(rows: list[dict[str, Any]] | None = None) -> AsyncMock:
    """Create a mock database context manager that returns configurable rows."""
    db = AsyncMock()
    cursor = AsyncMock()
    if rows is not None:
        fake_rows = [FakeRow(r) for r in rows]
        cursor.fetchone = AsyncMock(return_value=fake_rows[0] if fake_rows else None)
        cursor.fetchall = AsyncMock(return_value=fake_rows)
    else:
        cursor.fetchone = AsyncMock(return_value=None)
        cursor.fetchall = AsyncMock(return_value=[])
    db.execute = AsyncMock(return_value=cursor)
    db.commit = AsyncMock()
    return db


# ---------------------------------------------------------------------------
# Tests: _load_agent_row
# ---------------------------------------------------------------------------


class TestLoadAgentRow:
    async def test_loads_existing_agent(self):
        mock_db = _make_mock_db([FAKE_AGENT_ROW])

        with patch("orbiter_web.services.agent_runtime.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await _load_agent_row("agent-001")

        assert result["id"] == "agent-001"
        assert result["name"] == "Test Agent"
        assert result["model_provider"] == "openai"

    async def test_raises_for_missing_agent(self):
        mock_db = _make_mock_db(None)

        with patch("orbiter_web.services.agent_runtime.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(AgentRuntimeError, match="Agent not found"):
                await _load_agent_row("nonexistent")


# ---------------------------------------------------------------------------
# Tests: _resolve_provider
# ---------------------------------------------------------------------------


class TestResolveProvider:
    async def test_resolves_provider_with_encrypted_key(self):
        mock_db = _make_mock_db([FAKE_PROVIDER_ROW])

        with (
            patch("orbiter_web.services.agent_runtime.get_db") as mock_get_db,
            patch("orbiter_web.services.agent_runtime.decrypt_api_key") as mock_decrypt,
            patch("orbiter_web.services.agent_runtime.get_provider") as mock_get_prov,
        ):
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_decrypt.return_value = "sk-real-key"
            mock_provider = MagicMock()
            mock_get_prov.return_value = mock_provider

            result = await _resolve_provider("openai", "gpt-4o", "user-001")

        assert result is mock_provider
        mock_get_prov.assert_called_once_with(
            "openai:gpt-4o", api_key="sk-real-key", base_url=None
        )

    async def test_raises_for_no_provider(self):
        mock_db = _make_mock_db(None)

        with patch("orbiter_web.services.agent_runtime.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(AgentRuntimeError, match="No openai provider configured"):
                await _resolve_provider("openai", "gpt-4o", "user-001")


# ---------------------------------------------------------------------------
# Tests: _resolve_tools
# ---------------------------------------------------------------------------


class TestResolveTools:
    async def test_resolves_tools_from_db(self):
        mock_db = _make_mock_db([FAKE_TOOL_ROW])

        with patch("orbiter_web.services.agent_runtime.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)

            tools = await _resolve_tools('["tool-001"]', "proj-001", "user-001")

        assert len(tools) == 1
        assert tools[0].name == "search_web"
        assert tools[0].description == "Search the web for information"
        assert "query" in tools[0].parameters.get("properties", {})

    async def test_empty_tools_json(self):
        tools = await _resolve_tools("[]", "proj-001", "user-001")
        assert tools == []

    async def test_invalid_tools_json(self):
        tools = await _resolve_tools("not-json", "proj-001", "user-001")
        assert tools == []


# ---------------------------------------------------------------------------
# Tests: AgentService
# ---------------------------------------------------------------------------


class TestAgentService:
    async def test_build_agent(self):
        svc = AgentService()

        with patch("orbiter_web.services.agent_runtime._load_agent_row") as mock_load:
            mock_load.return_value = FAKE_AGENT_ROW

            with patch("orbiter_web.services.agent_runtime._resolve_tools") as mock_tools:
                mock_tools.return_value = []

                agent = await svc.build_agent("agent-001")

        assert agent.name == "Test Agent"
        assert agent.model == "openai:gpt-4o"
        assert agent.instructions == "You are a helpful assistant."
        assert agent.max_steps == 5
        assert agent.temperature == 0.7
        assert agent.max_tokens == 1024

    async def test_build_agent_no_model_raises(self):
        svc = AgentService()
        no_model_row = {**FAKE_AGENT_ROW, "model_provider": "", "model_name": ""}

        with patch("orbiter_web.services.agent_runtime._load_agent_row") as mock_load:
            mock_load.return_value = no_model_row

            with pytest.raises(AgentRuntimeError, match="has no model configured"):
                await svc.build_agent("agent-001")

    async def test_run_agent(self):
        svc = AgentService()
        mock_agent = AsyncMock()
        mock_agent.run.return_value = AgentOutput(
            text="Hello there!",
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

        mock_provider = MagicMock()

        with (
            patch("orbiter_web.services.agent_runtime._load_agent_row") as mock_load,
            patch.object(svc, "build_agent") as mock_build,
            patch("orbiter_web.services.agent_runtime._resolve_provider") as mock_resolve,
        ):
            mock_load.return_value = FAKE_AGENT_ROW
            mock_build.return_value = mock_agent
            mock_resolve.return_value = mock_provider

            messages = [UserMessage(content="Hello")]
            result = await svc.run_agent("agent-001", messages)

        assert result.content == "Hello there!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    async def test_run_agent_no_model_raises(self):
        svc = AgentService()
        no_model_row = {**FAKE_AGENT_ROW, "model_provider": "", "model_name": ""}

        with patch("orbiter_web.services.agent_runtime._load_agent_row") as mock_load:
            mock_load.return_value = no_model_row

            with pytest.raises(AgentRuntimeError, match="has no model configured"):
                await svc.run_agent("agent-001", [UserMessage(content="Hi")])

    async def test_run_agent_model_failure(self):
        svc = AgentService()
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = RuntimeError("API timeout")

        mock_provider = MagicMock()

        with (
            patch("orbiter_web.services.agent_runtime._load_agent_row") as mock_load,
            patch.object(svc, "build_agent") as mock_build,
            patch("orbiter_web.services.agent_runtime._resolve_provider") as mock_resolve,
        ):
            mock_load.return_value = FAKE_AGENT_ROW
            mock_build.return_value = mock_agent
            mock_resolve.return_value = mock_provider

            with pytest.raises(AgentRuntimeError, match="Model call failed"):
                await svc.run_agent("agent-001", [UserMessage(content="Hi")])

    async def test_stream_agent(self):
        svc = AgentService()

        chunks = [
            StreamChunk(delta="Hello"),
            StreamChunk(delta=" world"),
            StreamChunk(delta="!", finish_reason="stop"),
        ]

        async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[StreamChunk]:
            for c in chunks:
                yield c

        mock_provider = MagicMock()
        mock_provider.stream = fake_stream

        with (
            patch("orbiter_web.services.agent_runtime._load_agent_row") as mock_load,
            patch.object(svc, "build_agent") as mock_build,
            patch("orbiter_web.services.agent_runtime._resolve_provider") as mock_resolve,
        ):
            mock_load.return_value = FAKE_AGENT_ROW
            mock_build.return_value = MagicMock(
                instructions="You are helpful.",
                name="Test Agent",
                temperature=0.7,
                max_tokens=1024,
                get_tool_schemas=MagicMock(return_value=[]),
            )
            mock_resolve.return_value = mock_provider

            collected: list[StreamChunk] = []
            async for chunk in svc.stream_agent(
                "agent-001", [UserMessage(content="Hi")]
            ):
                collected.append(chunk)

        assert len(collected) == 3
        assert collected[0].delta == "Hello"
        assert collected[1].delta == " world"
        assert collected[2].finish_reason == "stop"

    async def test_stream_agent_no_model_raises(self):
        svc = AgentService()
        no_model_row = {**FAKE_AGENT_ROW, "model_provider": "", "model_name": ""}

        with patch("orbiter_web.services.agent_runtime._load_agent_row") as mock_load:
            mock_load.return_value = no_model_row

            with pytest.raises(AgentRuntimeError, match="has no model configured"):
                async for _ in svc.stream_agent(
                    "agent-001", [UserMessage(content="Hi")]
                ):
                    pass  # pragma: no cover
