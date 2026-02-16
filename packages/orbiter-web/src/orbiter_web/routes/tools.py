"""Tools REST API.

Manages user-defined tools stored in the database.
Also exposes built-in tools merged into listings.
"""

from __future__ import annotations

import ast
import json
import traceback
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/tools", tags=["tools"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {"search", "code", "file", "data", "communication", "custom"}
VALID_TOOL_TYPES = {"function", "http", "schema", "mcp"}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ToolResponse(BaseModel):
    id: str
    name: str
    description: str
    category: str
    schema_json: str
    code: str
    tool_type: str
    usage_count: int
    project_id: str
    user_id: str
    created_at: str


class ToolCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    project_id: str = Field(..., min_length=1)
    description: str = ""
    category: str = "custom"
    schema_json: str = "{}"
    code: str = ""
    tool_type: str = "function"


class ToolUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    category: str | None = None
    schema_json: str | None = None
    code: str | None = None
    tool_type: str | None = None


class CustomToolCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    project_id: str = Field(..., min_length=1)
    code: str = Field(..., min_length=1)
    description: str = ""


class CustomToolUpdate(BaseModel):
    code: str = Field(..., min_length=1)
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None


class CustomToolTestRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)


class ParameterInfo(BaseModel):
    name: str
    type: str
    default: str | None = None
    description: str = ""


class CustomToolSchemaResponse(BaseModel):
    tool: ToolResponse
    parameters: list[ParameterInfo]
    schema: dict[str, Any]


class CustomToolTestResponse(BaseModel):
    success: bool
    output: Any = None
    error: str | None = None
    traceback: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_ownership(db: Any, tool_id: str, user_id: str) -> dict[str, Any]:
    """Verify tool exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM tools WHERE id = ? AND user_id = ?",
        (tool_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# AST-based parameter extraction and schema generation
# ---------------------------------------------------------------------------

_AST_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def _annotation_to_json_type(node: ast.expr | None) -> str:
    """Convert an AST annotation node to a JSON Schema type string."""
    if node is None:
        return "string"
    if isinstance(node, ast.Name):
        return _AST_TYPE_MAP.get(node.id, "string")
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _AST_TYPE_MAP.get(node.value, "string")
    if isinstance(node, ast.Attribute):
        return "string"
    if isinstance(node, ast.Subscript):
        # Handle list[X], dict[K, V], Optional[X]
        if isinstance(node.value, ast.Name):
            if node.value.id in ("list", "List"):
                return "array"
            if node.value.id in ("dict", "Dict"):
                return "object"
            if node.value.id == "Optional":
                return _annotation_to_json_type(node.slice)
        return "string"
    return "string"


def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    """Parse Google-style Args: section from a docstring into {param: description}."""
    if not docstring:
        return {}
    params: dict[str, str] = {}
    in_args = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped[0].isspace() and ":" not in stripped:
                # Left-aligned non-param line — end of Args section
                break
            if stripped.startswith("returns:") or stripped.startswith("raises:"):
                break
            if ":" in stripped:
                parts = stripped.split(":", 1)
                param_part = parts[0].strip()
                # Handle "param_name (type)" or just "param_name"
                param_name = param_part.split("(")[0].strip()
                if param_name:
                    params[param_name] = parts[1].strip()
    return params


def _extract_function_info(
    code: str,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, str | None]:
    """Parse code and find the first decorated function (with @tool) or the first function."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None

    # Prefer the first @tool-decorated function, fall back to first function
    first_fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if first_fn is None:
                first_fn = node
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "tool":
                    docstring = ast.get_docstring(node)
                    return node, docstring
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "tool"
                ):
                    docstring = ast.get_docstring(node)
                    return node, docstring

    if first_fn is not None:
        return first_fn, ast.get_docstring(first_fn)
    return None, None


def _generate_schema_from_code(code: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate JSON schema and parameter list from Python tool code.

    Returns (schema_dict, parameters_list).
    """
    func_node, docstring = _extract_function_info(code)
    if func_node is None:
        raise HTTPException(
            status_code=422,
            detail="No function found in code. Define a function (optionally decorated with @tool).",
        )

    doc_params = _parse_docstring_params(docstring)

    properties: dict[str, Any] = {}
    required: list[str] = []
    parameters: list[dict[str, Any]] = []

    for arg in func_node.args.args:
        if arg.arg == "self":
            continue
        param_name = arg.arg
        json_type = _annotation_to_json_type(arg.annotation)
        param_desc = doc_params.get(param_name, "")

        prop: dict[str, Any] = {"type": json_type}
        if param_desc:
            prop["description"] = param_desc

        properties[param_name] = prop

        # Check for default value
        default_val: str | None = None
        # In AST, defaults are stored right-aligned: last N defaults match last N args
        num_defaults = len(func_node.args.defaults)
        num_args = len(func_node.args.args)
        arg_index = func_node.args.args.index(arg)
        default_index = arg_index - (num_args - num_defaults)

        if default_index >= 0:
            default_node = func_node.args.defaults[default_index]
            if isinstance(default_node, ast.Constant):
                default_val = repr(default_node.value)
            elif isinstance(default_node, ast.Name) and default_node.id == "None":
                default_val = "None"
        else:
            required.append(param_name)

        parameters.append(
            {
                "name": param_name,
                "type": json_type,
                "default": default_val,
                "description": param_desc,
            }
        )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    if docstring:
        # Use first line of docstring as description
        first_line = docstring.strip().split("\n")[0].strip()
        if first_line:
            schema["description"] = first_line

    return schema, parameters


def _execute_tool_code(code: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Execute tool code in a restricted namespace and return result or error."""
    func_node, _ = _extract_function_info(code)
    if func_node is None:
        return {"success": False, "error": "No function found in code", "traceback": None}

    func_name = func_node.name

    # Build a restricted namespace with safe builtins
    safe_builtins = (
        {
            k: v
            for k, v in __builtins__.items()  # type: ignore[union-attr]
            if k
            not in ("exec", "eval", "compile", "__import__", "open", "breakpoint", "exit", "quit")
        }
        if isinstance(__builtins__, dict)
        else {
            k: getattr(__builtins__, k)
            for k in dir(__builtins__)
            if k
            not in ("exec", "eval", "compile", "__import__", "open", "breakpoint", "exit", "quit")
        }
    )

    # Provide a no-op @tool decorator so user code doesn't error
    def _noop_tool(fn: Any = None, **_kwargs: Any) -> Any:
        if fn is not None:
            return fn
        return lambda f: f

    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "tool": _noop_tool}

    try:
        exec(compile(code, "<custom_tool>", "exec"), namespace)
    except Exception:
        return {
            "success": False,
            "error": "Failed to compile tool code",
            "traceback": traceback.format_exc(),
        }

    if func_name not in namespace:
        return {
            "success": False,
            "error": f"Function '{func_name}' not found after execution",
            "traceback": None,
        }

    func = namespace[func_name]
    try:
        result = func(**inputs)
        return {"success": True, "output": result}
    except Exception:
        return {
            "success": False,
            "error": "Tool execution failed",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Endpoints — custom tool routes (must be before /{tool_id} param routes)
# ---------------------------------------------------------------------------


@router.post("/custom", response_model=CustomToolSchemaResponse, status_code=201)
async def create_custom_tool(
    body: CustomToolCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a custom tool from Python code with auto-generated schema."""
    schema, parameters = _generate_schema_from_code(body.code)
    schema_json_str = json.dumps(schema)

    # Use description from body, or fall back to schema description
    description = body.description or schema.get("description", "")

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        tool_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO tools (
                id, name, description, category, schema_json, code,
                tool_type, usage_count, project_id, user_id, created_at
            ) VALUES (?, ?, ?, 'custom', ?, ?, 'function', 0, ?, ?, ?)
            """,
            (
                tool_id,
                body.name,
                description,
                schema_json_str,
                body.code,
                body.project_id,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return {
            "tool": _row_to_dict(row),
            "parameters": parameters,
            "schema": schema,
        }


@router.put("/custom/{tool_id}", response_model=CustomToolSchemaResponse)
async def update_custom_tool(
    tool_id: str,
    body: CustomToolUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a custom tool's code and regenerate schema."""
    schema, parameters = _generate_schema_from_code(body.code)
    schema_json_str = json.dumps(schema)

    async with get_db() as db:
        await _verify_ownership(db, tool_id, user["id"])

        updates: dict[str, str] = {
            "code": body.code,
            "schema_json": schema_json_str,
        }
        if body.name is not None:
            updates["name"] = body.name
        if body.description is not None:
            updates["description"] = body.description
        else:
            # Update description from schema if not explicitly provided
            desc = schema.get("description", "")
            if desc:
                updates["description"] = desc

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), tool_id]

        await db.execute(
            f"UPDATE tools SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return {
            "tool": _row_to_dict(row),
            "parameters": parameters,
            "schema": schema,
        }


@router.post("/custom/{tool_id}/test", response_model=CustomToolTestResponse)
async def test_custom_tool(
    tool_id: str,
    body: CustomToolTestRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Test a custom tool by executing it with sample inputs."""
    async with get_db() as db:
        tool_row = await _verify_ownership(db, tool_id, user["id"])

    code = tool_row["code"]
    if not code.strip():
        raise HTTPException(status_code=422, detail="Tool has no code to execute")

    return _execute_tool_code(code, body.inputs)


@router.post("/schema", response_model=dict[str, Any])
async def generate_schema(
    body: CustomToolUpdate,
    _user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Generate JSON schema from Python code without saving."""
    schema, parameters = _generate_schema_from_code(body.code)
    return {"schema": schema, "parameters": parameters}


# ---------------------------------------------------------------------------
# Endpoints — generic tool CRUD
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ToolResponse])
async def list_tools(
    category: str | None = Query(None),
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all tools for the current user, optionally filtered by category and/or project."""
    async with get_db() as db:
        conditions = ["user_id = ?"]
        params: list[str] = [user["id"]]

        if category:
            conditions.append("category = ?")
            params.append(category)
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        where = " AND ".join(conditions)
        cursor = await db.execute(
            f"SELECT * FROM tools WHERE {where} ORDER BY created_at DESC",
            params,
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.post("", response_model=ToolResponse, status_code=201)
async def create_tool(
    body: ToolCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new tool."""
    if body.category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}",
        )
    if body.tool_type not in VALID_TOOL_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid tool_type. Must be one of: {', '.join(sorted(VALID_TOOL_TYPES))}",
        )

    async with get_db() as db:
        # Verify project exists and belongs to user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        tool_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO tools (
                id, name, description, category, schema_json, code,
                tool_type, usage_count, project_id, user_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                tool_id,
                body.name,
                body.description,
                body.category,
                body.schema_json,
                body.code,
                body.tool_type,
                body.project_id,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(
    tool_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single tool by ID with full detail including schema."""
    async with get_db() as db:
        return await _verify_ownership(db, tool_id, user["id"])


@router.put("/{tool_id}", response_model=ToolResponse)
async def update_tool(
    tool_id: str,
    body: ToolUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a tool's editable fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    if "category" in updates and updates["category"] not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}",
        )
    if "tool_type" in updates and updates["tool_type"] not in VALID_TOOL_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid tool_type. Must be one of: {', '.join(sorted(VALID_TOOL_TYPES))}",
        )

    async with get_db() as db:
        await _verify_ownership(db, tool_id, user["id"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), tool_id]

        await db.execute(
            f"UPDATE tools SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{tool_id}", status_code=204)
async def delete_tool(
    tool_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a tool."""
    async with get_db() as db:
        await _verify_ownership(db, tool_id, user["id"])
        await db.execute("DELETE FROM tools WHERE id = ?", (tool_id,))
        await db.commit()
