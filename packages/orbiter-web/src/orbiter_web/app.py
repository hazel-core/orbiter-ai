"""FastAPI application for Orbiter Web."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from orbiter_web.config import settings
from orbiter_web.database import run_migrations
from orbiter_web.middleware.csrf import CSRFMiddleware
from orbiter_web.middleware.rate_limit import RateLimitMiddleware
from orbiter_web.routes.agents import router as agents_router
from orbiter_web.routes.applications import router as applications_router
from orbiter_web.routes.auth import router as auth_router
from orbiter_web.routes.conversations import router as conversations_router
from orbiter_web.routes.costs import router as costs_router
from orbiter_web.routes.crews import router as crews_router
from orbiter_web.routes.knowledge_bases import router as knowledge_bases_router
from orbiter_web.routes.models import router as models_router
from orbiter_web.routes.neuron_pipelines import router as neuron_pipelines_router
from orbiter_web.routes.plans import router as plans_router
from orbiter_web.routes.playground import router as playground_router
from orbiter_web.routes.plugins import router as plugins_router
from orbiter_web.routes.projects import router as projects_router
from orbiter_web.routes.prompt_templates import router as prompt_templates_router
from orbiter_web.routes.provider_keys import router as provider_keys_router
from orbiter_web.routes.providers import router as providers_router
from orbiter_web.routes.threads import router as threads_router
from orbiter_web.routes.tools import router as tools_router
from orbiter_web.routes.vector_stores import router as vector_stores_router
from orbiter_web.routes.workflow_runs import router as workflow_runs_router
from orbiter_web.routes.workflows import router as workflows_router
from orbiter_web.websocket import router as ws_router


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


app.add_middleware(CSRFMiddleware)
app.add_middleware(RateLimitMiddleware)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-CSRF-Token"],
    )

app.include_router(agents_router)
app.include_router(neuron_pipelines_router)
app.include_router(plans_router)
app.include_router(applications_router)
app.include_router(conversations_router)
app.include_router(costs_router)
app.include_router(crews_router)
app.include_router(knowledge_bases_router)
app.include_router(auth_router)
app.include_router(models_router)
app.include_router(playground_router)
app.include_router(plugins_router)
app.include_router(projects_router)
app.include_router(prompt_templates_router)
app.include_router(provider_keys_router)
app.include_router(providers_router)
app.include_router(threads_router)
app.include_router(tools_router)
app.include_router(vector_stores_router)
app.include_router(workflow_runs_router)
app.include_router(workflows_router)
app.include_router(ws_router)


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
