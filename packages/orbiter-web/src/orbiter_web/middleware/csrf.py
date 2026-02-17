"""CSRF protection middleware for cookie-based authentication."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from orbiter_web.database import get_db

# Methods that mutate state and require CSRF validation.
_UNSAFE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

# Paths exempt from CSRF checks (login must work without a token).
_EXEMPT_PATHS = {
    "/api/v1/auth/login",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/reset-password",
    "/api/health",
}

# Path prefixes exempt from CSRF (CI endpoints use API key auth, not cookies).
_EXEMPT_PREFIXES = ("/api/v1/ci/",)


class CSRFMiddleware(BaseHTTPMiddleware):
    """Validate X-CSRF-Token header on state-changing requests.

    The token is compared against the csrf_token stored in the user's session
    row. GET/HEAD/OPTIONS requests and exempt paths are always allowed through.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method not in _UNSAFE_METHODS:
            return await call_next(request)

        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        if any(request.url.path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)

        # WebSocket upgrades don't need CSRF.
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        session_id = request.cookies.get("orbiter_session")
        if not session_id:
            # No session cookie → auth middleware will reject later.
            return await call_next(request)

        csrf_header = request.headers.get("x-csrf-token", "")
        if not csrf_header:
            return JSONResponse(
                status_code=403,
                content={"detail": "Missing CSRF token"},
            )

        async with get_db() as db:
            cursor = await db.execute(
                "SELECT csrf_token FROM sessions WHERE id = ? AND expires_at > datetime('now')",
                (session_id,),
            )
            row = await cursor.fetchone()

        if row is None or row["csrf_token"] is None:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid session"},
            )

        if csrf_header != row["csrf_token"]:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid CSRF token"},
            )

        return await call_next(request)
