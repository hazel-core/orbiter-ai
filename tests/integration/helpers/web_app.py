"""Minimal FastAPI test application for integration tests.

Provides a /health endpoint used by the uvicorn_server fixture.
Additional routes are provided for agent-related integration tests.

Run standalone: uvicorn tests.integration.helpers.web_app:app
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Orbiter Integration Test App")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
