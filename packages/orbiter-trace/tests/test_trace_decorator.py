"""Tests for the @traced decorator, span context managers, and metadata extraction."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from orbiter.trace.decorator import (  # pyright: ignore[reportMissingImports]
    extract_metadata,
    get_user_frame,
    is_user_code,
    span_async,
    span_sync,
    traced,
)

# ---------------------------------------------------------------------------
# In-memory exporter (OTel SDK version may not ship InMemorySpanExporter)
# ---------------------------------------------------------------------------


class _MemoryExporter(SpanExporter):
    """Collects finished spans in a list for test assertions."""

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []

    def export(self, spans: Any) -> SpanExportResult:
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def get_finished_spans(self) -> list[ReadableSpan]:
        return list(self._spans)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _otel_setup() -> Any:
    """Set up an in-memory OTel provider for each test."""
    # Reset the global singleton guard so we can install a fresh provider.
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]

    exporter = _MemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    provider.shutdown()


@pytest.fixture()
def exporter(_otel_setup: Any) -> _MemoryExporter:
    return _otel_setup


# ---------------------------------------------------------------------------
# is_user_code
# ---------------------------------------------------------------------------


class TestIsUserCode:
    def test_normal_file(self, tmp_path: Any) -> None:
        assert is_user_code(str(tmp_path / "my_app.py")) is True

    def test_dynamic_code(self) -> None:
        assert is_user_code("<string>") is False

    def test_eval_code(self) -> None:
        assert is_user_code("<module>") is False

    def test_trace_module_filtered(self) -> None:
        import orbiter.trace.decorator as mod  # pyright: ignore[reportMissingImports]

        assert is_user_code(mod.__file__) is False


# ---------------------------------------------------------------------------
# get_user_frame
# ---------------------------------------------------------------------------


class TestGetUserFrame:
    def test_returns_frame_info(self) -> None:
        frame = get_user_frame()
        assert frame is not None
        assert "test_trace_decorator" in frame.filename

    def test_lineno_is_positive(self) -> None:
        frame = get_user_frame()
        assert frame is not None
        assert frame.lineno > 0

    def test_function_name(self) -> None:
        frame = get_user_frame()
        assert frame is not None
        assert "test_function_name" in frame.function


# ---------------------------------------------------------------------------
# extract_metadata
# ---------------------------------------------------------------------------


def _sample_func(x: int, y: str = "hello") -> str:
    return f"{x}-{y}"


class _SampleClass:
    def method(self, a: int) -> int:
        return a


class TestExtractMetadata:
    def test_qualname(self) -> None:
        meta = extract_metadata(_sample_func)
        assert meta["code.function"] == "_sample_func"

    def test_module(self) -> None:
        meta = extract_metadata(_sample_func)
        assert "test_trace_decorator" in meta["code.module"]

    def test_lineno(self) -> None:
        meta = extract_metadata(_sample_func)
        assert isinstance(meta["code.lineno"], int)
        assert meta["code.lineno"] > 0

    def test_filepath(self) -> None:
        meta = extract_metadata(_sample_func)
        assert "code.filepath" in meta

    def test_parameters(self) -> None:
        meta = extract_metadata(_sample_func)
        assert meta["code.parameters"] == ["x", "y"]

    def test_method_excludes_self(self) -> None:
        meta = extract_metadata(_SampleClass.method)
        assert "self" not in meta["code.parameters"]
        assert "a" in meta["code.parameters"]

    def test_lambda(self) -> None:
        fn = lambda x: x  # noqa: E731
        meta = extract_metadata(fn)
        assert "lambda" in meta["code.function"]

    def test_builtin_graceful(self) -> None:
        meta = extract_metadata(len)
        assert meta["code.function"] == "len"
        # builtins may or may not expose parameters depending on Python version
        assert isinstance(meta["code.parameters"], list)


# ---------------------------------------------------------------------------
# span_sync / span_async
# ---------------------------------------------------------------------------


class TestSpanSync:
    def test_creates_span(self, exporter: _MemoryExporter) -> None:
        with span_sync("test-span") as s:
            s.set_attribute("key", "val")
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test-span"

    def test_attributes(self, exporter: _MemoryExporter) -> None:
        with span_sync("attr-span", attributes={"custom": "value"}):
            pass
        spans = exporter.get_finished_spans()
        assert spans[0].attributes is not None
        assert spans[0].attributes.get("custom") == "value"

    def test_records_exception(self, exporter: _MemoryExporter) -> None:
        with pytest.raises(ValueError, match="boom"), span_sync("err-span"):
            raise ValueError("boom")
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        events = spans[0].events
        assert any(e.name == "exception" for e in events)


class TestSpanAsync:
    async def test_creates_span(self, exporter: _MemoryExporter) -> None:
        async with span_async("async-span") as s:
            s.set_attribute("async_key", 42)
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "async-span"

    async def test_records_exception(self, exporter: _MemoryExporter) -> None:
        with pytest.raises(RuntimeError, match="async boom"):
            async with span_async("async-err"):
                raise RuntimeError("async boom")
        spans = exporter.get_finished_spans()
        assert any(e.name == "exception" for e in spans[0].events)


# ---------------------------------------------------------------------------
# @traced — sync functions
# ---------------------------------------------------------------------------


class TestTracedSync:
    def test_basic_sync(self, exporter: _MemoryExporter) -> None:
        @traced()
        def add(a: int, b: int) -> int:
            return a + b

        result = add(1, 2)
        assert result == 3
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert "add" in spans[0].name

    def test_custom_name(self, exporter: _MemoryExporter) -> None:
        @traced(name="custom-op")
        def noop() -> None:
            pass

        noop()
        spans = exporter.get_finished_spans()
        assert spans[0].name == "custom-op"

    def test_extra_attributes(self, exporter: _MemoryExporter) -> None:
        @traced(attributes={"component": "test"})
        def work() -> str:
            return "done"

        work()
        spans = exporter.get_finished_spans()
        assert spans[0].attributes is not None
        assert spans[0].attributes.get("component") == "test"

    def test_extract_args(self, exporter: _MemoryExporter) -> None:
        @traced(extract_args=True)
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}"

        greet("World")
        spans = exporter.get_finished_spans()
        attrs = spans[0].attributes or {}
        assert attrs.get("arg.name") == "World"
        assert attrs.get("arg.greeting") == "Hello"

    def test_metadata_attributes(self, exporter: _MemoryExporter) -> None:
        @traced()
        def meta_func() -> None:
            pass

        meta_func()
        spans = exporter.get_finished_spans()
        attrs = spans[0].attributes or {}
        assert "code.function" in attrs
        assert "code.module" in attrs

    def test_exception_recorded(self, exporter: _MemoryExporter) -> None:
        @traced()
        def fail() -> None:
            raise ValueError("sync fail")

        with pytest.raises(ValueError, match="sync fail"):
            fail()
        spans = exporter.get_finished_spans()
        assert any(e.name == "exception" for e in spans[0].events)

    def test_preserves_metadata(self) -> None:
        @traced()
        def documented() -> None:
            """A documented function."""

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "A documented function."


# ---------------------------------------------------------------------------
# @traced — async functions
# ---------------------------------------------------------------------------


class TestTracedAsync:
    async def test_basic_async(self, exporter: _MemoryExporter) -> None:
        @traced()
        async def async_add(a: int, b: int) -> int:
            return a + b

        result = await async_add(3, 4)
        assert result == 7
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert "async_add" in spans[0].name

    async def test_async_exception(self, exporter: _MemoryExporter) -> None:
        @traced()
        async def async_fail() -> None:
            raise RuntimeError("async fail")

        with pytest.raises(RuntimeError, match="async fail"):
            await async_fail()
        spans = exporter.get_finished_spans()
        assert any(e.name == "exception" for e in spans[0].events)

    async def test_async_extract_args(self, exporter: _MemoryExporter) -> None:
        @traced(extract_args=True)
        async def fetch(url: str, timeout: int = 30) -> str:
            return url

        await fetch("http://example.com", timeout=5)
        attrs = exporter.get_finished_spans()[0].attributes or {}
        assert attrs.get("arg.url") == "http://example.com"
        assert attrs.get("arg.timeout") == "5"


# ---------------------------------------------------------------------------
# @traced — generators
# ---------------------------------------------------------------------------


class TestTracedGenerator:
    def test_sync_generator(self, exporter: _MemoryExporter) -> None:
        @traced()
        def gen_range(n: int) -> Any:
            yield from range(n)

        result = list(gen_range(3))
        assert result == [0, 1, 2]
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert "gen_range" in spans[0].name

    def test_generator_exception(self, exporter: _MemoryExporter) -> None:
        @traced()
        def gen_fail() -> Any:
            yield 1
            raise ValueError("gen fail")

        gen = gen_fail()
        assert next(gen) == 1
        with pytest.raises(ValueError, match="gen fail"):
            next(gen)
        spans = exporter.get_finished_spans()
        assert any(e.name == "exception" for e in spans[0].events)


class TestTracedAsyncGenerator:
    async def test_async_generator(self, exporter: _MemoryExporter) -> None:
        @traced()
        async def async_gen(n: int) -> Any:
            for i in range(n):
                yield i

        result = [item async for item in async_gen(3)]
        assert result == [0, 1, 2]
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert "async_gen" in spans[0].name

    async def test_async_generator_exception(self, exporter: _MemoryExporter) -> None:
        @traced()
        async def async_gen_fail() -> Any:
            yield 1
            raise RuntimeError("agen fail")

        gen = async_gen_fail()
        val = await gen.__anext__()
        assert val == 1
        with pytest.raises(RuntimeError, match="agen fail"):
            await gen.__anext__()
        spans = exporter.get_finished_spans()
        assert any(e.name == "exception" for e in spans[0].events)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestTracedEdgeCases:
    def test_nested_traced(self, exporter: _MemoryExporter) -> None:
        @traced()
        def outer() -> str:
            return inner()

        @traced()
        def inner() -> str:
            return "nested"

        assert outer() == "nested"
        spans = exporter.get_finished_spans()
        assert len(spans) == 2
        names = {s.name for s in spans}
        assert "outer" in str(names)
        assert "inner" in str(names)

    def test_list_attribute_flattened(self, exporter: _MemoryExporter) -> None:
        @traced(extract_args=True)
        def takes_list(items: list[int]) -> int:
            return sum(items)

        takes_list([1, 2, 3])
        attrs = exporter.get_finished_spans()[0].attributes or {}
        assert isinstance(attrs.get("arg.items"), str)

    async def test_async_with_custom_name_and_attrs(self, exporter: _MemoryExporter) -> None:
        @traced(name="custom-async", attributes={"tier": "premium"})
        async def premium_op() -> str:
            return "ok"

        await premium_op()
        span = exporter.get_finished_spans()[0]
        assert span.name == "custom-async"
        assert span.attributes is not None
        assert span.attributes.get("tier") == "premium"
