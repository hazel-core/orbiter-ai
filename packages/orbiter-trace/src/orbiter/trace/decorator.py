"""@traced decorator and span context manager for function-level instrumentation."""

from __future__ import annotations

import asyncio
import functools
import inspect
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from opentelemetry import trace

P = ParamSpec("P")
T = TypeVar("T")

# ---------------------------------------------------------------------------
# User-code filtering
# ---------------------------------------------------------------------------

_NON_USER_PREFIXES: tuple[str, ...] = (
    str(Path(inspect.__file__).parent),
    str(Path(__file__).parent),
)


def is_user_code(filename: str) -> bool:
    """Return True if *filename* belongs to user code (not stdlib/trace)."""
    if filename.startswith("<"):
        return False
    abs_path = str(Path(filename).resolve())
    return not any(abs_path.startswith(p) for p in _NON_USER_PREFIXES)


def get_user_frame() -> inspect.FrameInfo | None:
    """Walk the call stack to find the first user-code frame."""
    frame = sys._getframe(1)
    while frame is not None:
        if is_user_code(frame.f_code.co_filename):
            return inspect.FrameInfo(
                frame,
                frame.f_code.co_filename,
                frame.f_lineno,
                frame.f_code.co_qualname,
                None,
                None,
            )
        frame = frame.f_back
    return None


# ---------------------------------------------------------------------------
# Function metadata extraction
# ---------------------------------------------------------------------------


def extract_metadata(func: Any) -> dict[str, Any]:
    """Extract tracing metadata from a callable."""
    attrs: dict[str, Any] = {}
    code = getattr(func, "__code__", None)
    attrs["code.function"] = getattr(func, "__qualname__", getattr(func, "__name__", str(func)))
    attrs["code.module"] = getattr(func, "__module__", "")
    if code is not None:
        attrs["code.lineno"] = code.co_firstlineno
        filepath = code.co_filename
        try:
            attrs["code.filepath"] = str(Path(filepath).relative_to(Path.cwd()))
        except ValueError:
            attrs["code.filepath"] = filepath
    try:
        sig = inspect.signature(func)
        attrs["code.parameters"] = [p for p in sig.parameters if p != "self"]
    except (ValueError, TypeError):
        attrs["code.parameters"] = []
    return attrs


# ---------------------------------------------------------------------------
# Span context managers
# ---------------------------------------------------------------------------


@contextmanager
def span_sync(name: str, attributes: dict[str, Any] | None = None) -> Iterator[trace.Span]:
    """Synchronous span context manager."""
    tracer = trace.get_tracer("orbiter")
    with tracer.start_as_current_span(name, attributes=attributes or {}) as s:
        yield s


@asynccontextmanager
async def span_async(
    name: str, attributes: dict[str, Any] | None = None
) -> AsyncIterator[trace.Span]:
    """Asynchronous span context manager."""
    tracer = trace.get_tracer("orbiter")
    with tracer.start_as_current_span(name, attributes=attributes or {}) as s:
        yield s


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------


def traced(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
    extract_args: bool = False,
) -> Any:
    """Decorator that wraps a function in an OpenTelemetry span.

    Supports sync functions, async functions, sync generators, and async
    generators.  Metadata (qualname, module, line number, parameters) is
    automatically recorded as span attributes.

    Args:
        name: Span name override (defaults to ``func.__qualname__``).
        attributes: Extra attributes merged onto the span.
        extract_args: When *True*, record the function's call arguments.
    """

    def decorator(func: Any) -> Any:
        meta = extract_metadata(func)
        span_name = name or meta["code.function"]

        def _build_attrs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
            merged = dict(meta)
            if attributes:
                merged.update(attributes)
            if extract_args:
                try:
                    bound = inspect.signature(func).bind(*args, **kwargs)
                    bound.apply_defaults()
                    for k, v in bound.arguments.items():
                        if k != "self":
                            merged[f"arg.{k}"] = str(v)
                except TypeError:
                    pass
            # Flatten list values to strings for OTel compatibility.
            return {k: (str(v) if isinstance(v, list) else v) for k, v in merged.items()}

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = trace.get_tracer("orbiter")
                with tracer.start_as_current_span(
                    span_name, attributes=_build_attrs(args, kwargs)
                ) as s:
                    try:
                        async for item in func(*args, **kwargs):
                            yield item
                    except BaseException as exc:
                        s.record_exception(exc)
                        raise

            return async_gen_wrapper

        if inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = trace.get_tracer("orbiter")
                with tracer.start_as_current_span(
                    span_name, attributes=_build_attrs(args, kwargs)
                ) as s:
                    try:
                        yield from func(*args, **kwargs)
                    except BaseException as exc:
                        s.record_exception(exc)
                        raise

            return gen_wrapper

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = trace.get_tracer("orbiter")
                with tracer.start_as_current_span(
                    span_name, attributes=_build_attrs(args, kwargs)
                ) as s:
                    try:
                        return await func(*args, **kwargs)
                    except BaseException as exc:
                        s.record_exception(exc)
                        raise

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer("orbiter")
            with tracer.start_as_current_span(
                span_name, attributes=_build_attrs(args, kwargs)
            ) as s:
                try:
                    return func(*args, **kwargs)
                except BaseException as exc:
                    s.record_exception(exc)
                    raise

        return sync_wrapper

    return decorator
