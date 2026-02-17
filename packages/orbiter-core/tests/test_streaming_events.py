"""Tests for ToolResultEvent streaming type (US-002)."""

import pytest
from pydantic import ValidationError

from orbiter.types import (
    StepEvent,
    StreamEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class TestToolResultEvent:
    def test_create_success(self) -> None:
        e = ToolResultEvent(
            tool_name="search",
            tool_call_id="tc_1",
            arguments={"query": "hello"},
            result="found it",
            success=True,
            duration_ms=42.5,
            agent_name="bot",
        )
        assert e.type == "tool_result"
        assert e.tool_name == "search"
        assert e.tool_call_id == "tc_1"
        assert e.arguments == {"query": "hello"}
        assert e.result == "found it"
        assert e.error is None
        assert e.success is True
        assert e.duration_ms == 42.5
        assert e.agent_name == "bot"

    def test_create_failure(self) -> None:
        e = ToolResultEvent(
            tool_name="search",
            tool_call_id="tc_1",
            arguments={"query": "hello"},
            result="",
            error="not found",
            success=False,
            duration_ms=10.0,
            agent_name="bot",
        )
        assert e.success is False
        assert e.error == "not found"
        assert e.result == ""

    def test_defaults(self) -> None:
        e = ToolResultEvent(tool_name="search", tool_call_id="tc_1")
        assert e.arguments == {}
        assert e.result == ""
        assert e.error is None
        assert e.success is True
        assert e.duration_ms == 0.0
        assert e.agent_name == ""

    def test_frozen(self) -> None:
        e = ToolResultEvent(tool_name="search", tool_call_id="tc_1")
        with pytest.raises(ValidationError):
            e.tool_name = "other"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        e = ToolResultEvent(
            tool_name="search",
            tool_call_id="tc_1",
            arguments={"q": "test"},
            result="ok",
            error=None,
            success=True,
            duration_ms=15.3,
            agent_name="bot",
        )
        data = e.model_dump()
        assert data == {
            "type": "tool_result",
            "tool_name": "search",
            "tool_call_id": "tc_1",
            "arguments": {"q": "test"},
            "result": "ok",
            "error": None,
            "success": True,
            "duration_ms": 15.3,
            "agent_name": "bot",
        }
        restored = ToolResultEvent.model_validate(data)
        assert restored == e

    def test_arguments_default_not_shared(self) -> None:
        a = ToolResultEvent(tool_name="x", tool_call_id="1")
        b = ToolResultEvent(tool_name="y", tool_call_id="2")
        assert a.arguments is not b.arguments

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ToolResultEvent()  # type: ignore[call-arg]

    def test_type_literal(self) -> None:
        e = ToolResultEvent(tool_name="search", tool_call_id="tc_1")
        assert e.type == "tool_result"


class TestStreamEventUnionWithToolResult:
    def test_tool_result_event_is_stream_event(self) -> None:
        e: StreamEvent = ToolResultEvent(tool_name="x", tool_call_id="tc_1")
        assert isinstance(e, ToolResultEvent)

    def test_all_current_event_types_in_union(self) -> None:
        events: list[StreamEvent] = [
            TextEvent(text="hi"),
            ToolCallEvent(tool_name="search", tool_call_id="tc_1"),
            StepEvent(
                step_number=1,
                agent_name="bot",
                status="started",
                started_at=1000.0,
            ),
            ToolResultEvent(
                tool_name="search",
                tool_call_id="tc_1",
                result="ok",
            ),
        ]
        assert events[0].type == "text"
        assert events[1].type == "tool_call"
        assert events[2].type == "step"
        assert events[3].type == "tool_result"

    def test_discriminate_by_type_field(self) -> None:
        events: list[StreamEvent] = [
            ToolResultEvent(tool_name="x", tool_call_id="tc_1"),
            TextEvent(text="hello"),
        ]
        types = [e.type for e in events]
        assert types == ["tool_result", "text"]
