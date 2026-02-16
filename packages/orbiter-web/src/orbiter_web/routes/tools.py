"""Tools catalog REST API.

Returns available built-in tools that can be assigned to agents.
Tools are hard-coded for now; a future user-defined tools table
will augment this list.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/tools", tags=["tools"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ToolResponse(BaseModel):
    id: str
    name: str
    description: str
    category: str


# ---------------------------------------------------------------------------
# Built-in tool catalog
# ---------------------------------------------------------------------------

BUILTIN_TOOLS: list[dict[str, str]] = [
    {
        "id": "web_search",
        "name": "Web Search",
        "description": "Search the web for information",
        "category": "retrieval",
    },
    {
        "id": "file_read",
        "name": "File Read",
        "description": "Read contents of a file",
        "category": "filesystem",
    },
    {
        "id": "file_write",
        "name": "File Write",
        "description": "Write or update a file",
        "category": "filesystem",
    },
    {
        "id": "code_interpreter",
        "name": "Code Interpreter",
        "description": "Execute Python code in a sandbox",
        "category": "execution",
    },
    {
        "id": "http_request",
        "name": "HTTP Request",
        "description": "Make HTTP requests to external APIs",
        "category": "retrieval",
    },
    {
        "id": "database_query",
        "name": "Database Query",
        "description": "Execute SQL queries against a database",
        "category": "data",
    },
    {
        "id": "knowledge_search",
        "name": "Knowledge Search",
        "description": "Search the RAG knowledge base",
        "category": "retrieval",
    },
    {
        "id": "calculator",
        "name": "Calculator",
        "description": "Perform mathematical calculations",
        "category": "utility",
    },
    {
        "id": "json_parser",
        "name": "JSON Parser",
        "description": "Parse and transform JSON data",
        "category": "utility",
    },
    {
        "id": "text_splitter",
        "name": "Text Splitter",
        "description": "Split text into chunks for processing",
        "category": "utility",
    },
]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ToolResponse])
async def list_tools() -> list[dict[str, str]]:
    """Return all available tools."""
    return BUILTIN_TOOLS
