"""FastAPI application for Orbiter Web."""

from __future__ import annotations

from fastapi import FastAPI

from orbiter_web.config import settings

app = FastAPI(
    title="Orbiter Web API",
    version="0.1.0",
    debug=settings.debug,
)


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
