"""Integration tests for web API agent run via REST.

Tests that:
- The full REST agent run flow works: register → login → create agent → run → response.
- The run response includes usage statistics with total_tokens > 0.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_full_rest_agent_run_flow(http_client, vertex_model: str) -> None:
    """Full REST flow: register → login → create agent → run → assert output contains '4'."""
    # Register a new user
    reg_resp = await http_client.post(
        "/api/auth/register",
        json={"username": "testuser_int022a", "password": "testpass123"},
    )
    assert reg_resp.status_code == 201

    # Login to get a Bearer token
    login_resp = await http_client.post(
        "/api/auth/login",
        json={"username": "testuser_int022a", "password": "testpass123"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create an agent with the Vertex model
    create_resp = await http_client.post(
        "/api/agents",
        json={"model": vertex_model},
        headers=headers,
    )
    assert create_resp.status_code == 201
    agent_id = create_resp.json()["id"]
    assert agent_id

    # Run the agent — override client timeout for the LLM call
    run_resp = await http_client.post(
        f"/api/agents/{agent_id}/run",
        json={"input": "What is 2+2? Respond with just the number."},
        headers=headers,
        timeout=55.0,
    )
    assert run_resp.status_code == 200
    data = run_resp.json()
    assert "output" in data, f"Response missing 'output' field: {data}"
    assert "4" in data["output"], f"Expected '4' in output, got: {data['output']!r}"


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_agent_run_returns_usage_stats(http_client, vertex_model: str) -> None:
    """Run response includes usage stats with total_tokens > 0."""
    # Register a new user
    reg_resp = await http_client.post(
        "/api/auth/register",
        json={"username": "testuser_int022b", "password": "testpass123"},
    )
    assert reg_resp.status_code == 201

    # Login
    login_resp = await http_client.post(
        "/api/auth/login",
        json={"username": "testuser_int022b", "password": "testpass123"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create agent
    create_resp = await http_client.post(
        "/api/agents",
        json={"model": vertex_model},
        headers=headers,
    )
    assert create_resp.status_code == 201
    agent_id = create_resp.json()["id"]

    # Run agent — override client timeout for the LLM call
    run_resp = await http_client.post(
        f"/api/agents/{agent_id}/run",
        json={"input": "What is 2+2? Respond with just the number."},
        headers=headers,
        timeout=55.0,
    )
    assert run_resp.status_code == 200
    data = run_resp.json()
    assert "usage" in data, f"Response missing 'usage' field: {data}"
    usage = data["usage"]
    assert "total_tokens" in usage, f"Usage missing 'total_tokens': {usage}"
    assert usage["total_tokens"] > 0, f"Expected total_tokens > 0, got: {usage}"
