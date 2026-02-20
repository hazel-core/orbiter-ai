"""Minimal FastAPI test application for integration tests.

Provides a /health endpoint used by the uvicorn_server fixture.
Additional routes are provided for agent-related integration tests.

Run standalone: uvicorn tests.integration.helpers.web_app:app
"""

from __future__ import annotations

import secrets
import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

app = FastAPI(title="Orbiter Integration Test App")

# ---------------------------------------------------------------------------
# In-memory stores (module-level, shared across requests in same process)
# ---------------------------------------------------------------------------

_users: dict[str, dict] = {}  # username → {"id": str, "password": str}
_tokens: dict[str, str] = {}  # token → username
_agents: dict[str, dict] = {}  # agent_id → config dict

security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
) -> str:
    """Validate Bearer token and return username.

    Returns 401 for both missing Authorization header and invalid tokens.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    username = _tokens.get(credentials.credentials)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return username


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateAgentRequest(BaseModel):
    model: str
    name: str = ""
    instructions: str = ""
    tools: list[str] = []


class RunAgentRequest(BaseModel):
    input: str
    conversation_id: str = ""


class StreamAgentRequest(BaseModel):
    input: str
    conversation_id: str = ""


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------


@app.post("/api/auth/register", status_code=201)
async def register(req: RegisterRequest) -> dict:
    """Register a new user. Returns 409 if username already taken."""
    if req.username in _users:
        raise HTTPException(status_code=409, detail="User already exists")
    user_id = str(uuid.uuid4())
    _users[req.username] = {"id": user_id, "password": req.password}
    return {"id": user_id, "username": req.username}


@app.post("/api/auth/login")
async def login(req: LoginRequest) -> dict:
    """Authenticate and return a Bearer token."""
    user = _users.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = secrets.token_urlsafe(32)
    _tokens[token] = req.username
    return {"token": token, "token_type": "bearer"}


# ---------------------------------------------------------------------------
# Agent management routes
# ---------------------------------------------------------------------------


@app.get("/api/agents")
async def list_agents(username: str = Depends(_require_auth)) -> list:
    """List all agents owned by the authenticated user."""
    return [a for a in _agents.values() if a["owner"] == username]


@app.post("/api/agents", status_code=201)
async def create_agent(
    req: CreateAgentRequest, username: str = Depends(_require_auth)
) -> dict:
    """Create a new agent config and return its id."""
    agent_id = str(uuid.uuid4())
    agent_name = req.name or f"agent-{agent_id[:8]}"
    _agents[agent_id] = {
        "id": agent_id,
        "model": req.model,
        "name": agent_name,
        "instructions": req.instructions,
        "owner": username,
        "tools": req.tools,
    }
    return _agents[agent_id]


@app.post("/api/agents/{agent_id}/run")
async def run_agent(
    agent_id: str,
    req: RunAgentRequest,
    username: str = Depends(_require_auth),
) -> dict:
    """Run an agent and return its text output and usage statistics."""
    agent_config = _agents.get(agent_id)
    if not agent_config:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent_config["owner"] != username:
        raise HTTPException(status_code=403, detail="Forbidden")

    from orbiter.agent import Agent  # pyright: ignore[reportMissingImports]
    from orbiter.models import get_provider  # pyright: ignore[reportMissingImports]

    provider = get_provider(agent_config["model"])
    agent = Agent(
        name=agent_config["name"],
        model=agent_config["model"],
        instructions=agent_config.get("instructions", ""),
    )

    run_kwargs: dict = {"provider": provider}
    if req.conversation_id:
        run_kwargs["conversation_id"] = req.conversation_id

    result = await agent.run(req.input, **run_kwargs)

    return {
        "output": result.text,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        },
    }


@app.post("/api/agents/{agent_id}/stream")
async def stream_agent(
    agent_id: str,
    req: StreamAgentRequest,
    username: str = Depends(_require_auth),
) -> StreamingResponse:
    """Stream an agent run as Server-Sent Events (SSE).

    Each event is emitted as ``data: {json}\\n\\n``.  Event JSON has a
    ``type`` field matching the orbiter stream event types: ``text``,
    ``tool_call``, ``usage``.
    """
    agent_config = _agents.get(agent_id)
    if not agent_config:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent_config["owner"] != username:
        raise HTTPException(status_code=403, detail="Forbidden")

    async def _event_generator():
        from orbiter.agent import Agent  # pyright: ignore[reportMissingImports]
        from orbiter.models import get_provider  # pyright: ignore[reportMissingImports]
        from orbiter.runner import run  # pyright: ignore[reportMissingImports]
        from orbiter.tool import tool  # pyright: ignore[reportMissingImports]

        @tool
        def get_greeting(name: str) -> str:
            """Return a greeting message for a person.

            Args:
                name: The name of the person to greet.
            """
            return f"Hello, {name}!"

        _tool_registry = {"get_greeting": get_greeting}
        provider = get_provider(agent_config["model"])
        agent_tools: list = [
            _tool_registry[t]
            for t in agent_config.get("tools", [])
            if t in _tool_registry
        ]
        agent = Agent(
            name=agent_config["name"],
            model=agent_config["model"],
            instructions=agent_config.get("instructions", ""),
            tools=agent_tools,
        )

        async for event in run.stream(  # type: ignore[attr-defined]
            agent,
            req.input,
            provider=provider,
            detailed=True,
            event_types={"text", "tool_call", "usage"},
        ):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")
