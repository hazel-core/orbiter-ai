"""Integration tests for Marathon — platform end-to-end test.

US-INT-030: Exercises the full platform stack:
  Web API → agent with MCP tool (large output) → workspace offload
  → retrieve_artifact → vector memory → structured REST response.

The test registers a user, logs in, creates an agent via REST with:
  - MCP server config pointing to mcp_test_server.py
  - large_output_tools=['get_large_dataset'] to trigger workspace offload
  - vector_memory_path for Chroma persistence

It then runs the agent and asserts:
  1. REST response has non-empty output field.
  2. workspace_artifact_count > 0 (large output was offloaded to workspace).
  3. retrieve_artifact appears in tool_calls (agent accessed the artifact).
  4. vector_memory_count > 0 (agent output persisted to vector memory).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

_HELPERS_DIR = Path(__file__).parent / "helpers"
_MCP_SERVER_SCRIPT = str(_HELPERS_DIR / "mcp_test_server.py")


@pytest.mark.integration
@pytest.mark.marathon
@pytest.mark.timeout(180)
async def test_full_platform_stack_produces_structured_response(
    uvicorn_server: str,
    mcp_server_process,
    http_client,
    redis_container: str,
    vertex_model: str,
    tmp_path: Path,
) -> None:
    """Full platform marathon: REST → MCP large output → workspace offload → vector memory.

    Flow:
    1. Register and login via REST.
    2. Create an agent via REST with MCP config (large_output_tools=['get_large_dataset'])
       and a vector_memory_path for Chroma persistence.
    3. Run the agent with a prompt forcing get_large_dataset call.
    4. Assert:
       - REST response has non-empty output field.
       - workspace_artifact_count > 0 (large output offloaded to workspace).
       - retrieve_artifact appears in tool_calls (agent accessed the artifact).
       - vector_memory_count > 0 (agent output saved to vector memory).
    """
    # Use the MCP server script path from the session fixture
    mcp_script_path = mcp_server_process.args[0]
    vector_memory_path = str(tmp_path / "chroma_vec")

    # --- Auth flow ---
    username = f"marathon030_{uuid.uuid4().hex[:8]}"
    password = "marathon_pass_030"

    reg_resp = await http_client.post(
        "/api/auth/register",
        json={"username": username, "password": password},
    )
    assert reg_resp.status_code == 201, f"Register failed: {reg_resp.text}"

    login_resp = await http_client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # --- Create agent with MCP config and vector memory path ---
    create_resp = await http_client.post(
        "/api/agents",
        json={
            "model": vertex_model,
            "name": "platform-marathon-agent",
            "instructions": (
                "You are a data retrieval assistant. "
                "When instructed to call a tool, call it immediately. "
                "If a tool result is stored as an artifact, call retrieve_artifact "
                "to access the full content before responding. "
                "After retrieving the data, provide a concise summary of the dataset."
            ),
            "mcp_server_script": mcp_script_path,
            "mcp_large_output_tools": ["get_large_dataset"],
            "vector_memory_path": vector_memory_path,
        },
        headers=headers,
    )
    assert create_resp.status_code == 201, f"Create agent failed: {create_resp.text}"
    agent_id = create_resp.json()["id"]
    assert agent_id, "Expected non-empty agent_id"

    # --- Run agent with prompt forcing get_large_dataset → retrieve_artifact flow ---
    run_resp = await http_client.post(
        f"/api/agents/{agent_id}/run",
        json={
            "input": (
                "You MUST call the get_large_dataset tool with topic=astronomy. "
                "If the result is stored as an artifact, immediately call "
                "retrieve_artifact to access the full content. "
                "After retrieving, provide a brief summary of what the dataset contains. "
                "Do not answer from memory — use the tools."
            )
        },
        headers=headers,
        timeout=150.0,
    )
    assert run_resp.status_code == 200, f"Agent run failed: {run_resp.text}"
    data = run_resp.json()

    # --- Assertion 1: REST response has non-empty output ---
    assert data.get("output"), (
        f"Expected non-empty output field in run response, got: {data}"
    )

    # --- Assertion 2: workspace artifact created (large output was offloaded) ---
    assert data.get("workspace_artifact_count", 0) > 0, (
        f"Expected workspace_artifact_count > 0 (large MCP output should be offloaded), "
        f"got: {data.get('workspace_artifact_count')}. Full response: {data}"
    )

    # --- Assertion 3: retrieve_artifact tool call in response messages ---
    tool_calls = data.get("tool_calls", [])
    assert any("retrieve_artifact" in tc.get("name", "") for tc in tool_calls), (
        f"Expected 'retrieve_artifact' in tool_calls after large output offload, "
        f"got: {[tc.get('name') for tc in tool_calls]}"
    )

    # --- Assertion 4: vector memory has entries from the run ---
    assert data.get("vector_memory_count", 0) > 0, (
        f"Expected vector_memory_count > 0 (agent output persisted to Chroma), "
        f"got: {data.get('vector_memory_count')}. Full response: {data}"
    )
