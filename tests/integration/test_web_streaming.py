"""Integration tests for web API SSE streaming endpoint.

Tests that:
- The SSE stream contains text, tool_call, and usage event types.
- The last event has type == 'usage' with total_tokens > 0.
- No error events appear in the stream during a successful run.
"""

from __future__ import annotations

import json

import pytest


async def _auth_and_create_streaming_agent(http_client, vertex_model: str, username: str) -> tuple[dict, str]:
    """Register, login, create an agent with get_greeting tool; return (headers, agent_id)."""
    reg_resp = await http_client.post(
        "/api/auth/register",
        json={"username": username, "password": "testpass123"},
    )
    assert reg_resp.status_code == 201

    login_resp = await http_client.post(
        "/api/auth/login",
        json={"username": username, "password": "testpass123"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    create_resp = await http_client.post(
        "/api/agents",
        json={
            "model": vertex_model,
            "instructions": "You are a helpful assistant. Always use your tools when asked.",
            "tools": ["get_greeting"],
        },
        headers=headers,
    )
    assert create_resp.status_code == 201
    agent_id = create_resp.json()["id"]
    return headers, agent_id


async def _consume_sse_stream(http_client, agent_id: str, headers: dict, prompt: str) -> list[dict]:
    """POST to /api/agents/{id}/stream and collect all SSE events as dicts."""
    events: list[dict] = []
    async with http_client.stream(
        "POST",
        f"/api/agents/{agent_id}/stream",
        json={"input": prompt},
        headers=headers,
        timeout=55.0,
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_sse_stream_contains_required_event_types(http_client, vertex_model: str) -> None:
    """SSE stream emits text, tool_call, and usage events in correct order.

    Uses get_greeting tool with a constrained prompt so that:
    - At least one tool_call event is emitted (step 1: tool call)
    - At least one text event is emitted (step 2: final response)
    - Last event is usage with total_tokens > 0
    """
    headers, agent_id = await _auth_and_create_streaming_agent(
        http_client, vertex_model, "testuser_int023a"
    )
    prompt = (
        "You MUST call the get_greeting tool with name='World'. "
        "Do not respond without calling the tool first."
    )
    events = await _consume_sse_stream(http_client, agent_id, headers, prompt)

    assert events, "No SSE events received from the stream"

    event_types = [e["type"] for e in events]
    assert "text" in event_types, f"No 'text' event in stream. Events: {event_types}"
    assert "tool_call" in event_types, f"No 'tool_call' event in stream. Events: {event_types}"
    assert "usage" in event_types, f"No 'usage' event in stream. Events: {event_types}"

    last = events[-1]
    assert last["type"] == "usage", (
        f"Expected last event type 'usage', got '{last['type']}'. Events: {event_types}"
    )
    assert last["usage"]["total_tokens"] > 0, (
        f"Expected total_tokens > 0 in last usage event, got: {last['usage']}"
    )


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_sse_stream_no_error_events(http_client, vertex_model: str) -> None:
    """SSE stream produces no error events during a successful agent run."""
    headers, agent_id = await _auth_and_create_streaming_agent(
        http_client, vertex_model, "testuser_int023b"
    )
    prompt = (
        "You MUST call the get_greeting tool with name='World'. "
        "Do not respond without calling the tool first."
    )
    events = await _consume_sse_stream(http_client, agent_id, headers, prompt)

    assert events, "No SSE events received from the stream"

    error_events = [e for e in events if e.get("type") == "error"]
    assert not error_events, f"Unexpected error event(s) in stream: {error_events}"
