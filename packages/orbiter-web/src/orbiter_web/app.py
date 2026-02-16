"""FastAPI application for Orbiter Web."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orbiter_web.config import settings
from orbiter_web.database import run_migrations
from orbiter_web.routes.auth import router as auth_router
from orbiter_web.routes.projects import router as projects_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run migrations on startup."""
    await run_migrations()
    yield


app = FastAPI(
    title="Orbiter Web API",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)


app.include_router(auth_router)
app.include_router(projects_router)


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
