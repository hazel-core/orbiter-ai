"""Integration tests for web API authentication flow.

Tests that:
- Valid Bearer token allows access to protected routes.
- Invalid Bearer token is rejected with 401.
- Missing Authorization header is rejected with 401.
- Login with wrong password is rejected with 401 or 403.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_valid_jwt_accesses_protected_route(http_client) -> None:
    """Register → login → GET /api/agents with valid token → 200."""
    # Register a new user
    reg_resp = await http_client.post(
        "/api/auth/register",
        json={"username": "testuser_int024a", "password": "validpass123"},
    )
    assert reg_resp.status_code == 201

    # Login to obtain a Bearer token
    login_resp = await http_client.post(
        "/api/auth/login",
        json={"username": "testuser_int024a", "password": "validpass123"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["token"]
    assert token

    # Access a protected route with the valid token → 200
    list_resp = await http_client.get(
        "/api/agents",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_resp.status_code == 200


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_invalid_jwt_rejected(http_client) -> None:
    """GET /api/agents with a garbage token → 401."""
    list_resp = await http_client.get(
        "/api/agents",
        headers={"Authorization": "Bearer garbage.token.here"},
    )
    assert list_resp.status_code == 401


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_missing_auth_rejected(http_client) -> None:
    """GET /api/agents with no Authorization header → 401."""
    list_resp = await http_client.get("/api/agents")
    assert list_resp.status_code == 401


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_password_wrong_login_rejected(http_client) -> None:
    """Register a user then attempt login with wrong password → 401 or 403."""
    # Register a new user
    reg_resp = await http_client.post(
        "/api/auth/register",
        json={"username": "testuser_int024b", "password": "correctpass"},
    )
    assert reg_resp.status_code == 201

    # Login with the wrong password — must be rejected
    login_resp = await http_client.post(
        "/api/auth/login",
        json={"username": "testuser_int024b", "password": "wrongpassword"},
    )
    assert login_resp.status_code in (401, 403)
