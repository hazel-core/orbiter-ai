"""Tests for VectorMemoryStore and Embeddings ABC."""

from __future__ import annotations

import math

import pytest

from orbiter.memory.backends.vector import (  # pyright: ignore[reportMissingImports]
    Embeddings,
    OpenAIEmbeddings,
    VectorMemoryStore,
    _cosine_similarity,
)
from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
    HumanMemory,
    MemoryMetadata,
    MemoryStatus,
    MemoryStore,
    SystemMemory,
)

# ---------------------------------------------------------------------------
# Mock embeddings — deterministic vectors for testing
# ---------------------------------------------------------------------------


class MockEmbeddings(Embeddings):
    """Deterministic embeddings for testing.

    Generates a simple vector based on the character-code average of the text.
    """

    __slots__ = ("_dim", "call_count")

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        if not text:
            return [0.0] * self._dim
        # Simple deterministic embedding: use char codes
        base = sum(ord(c) for c in text) / len(text) / 256.0
        return [base + i * 0.1 for i in range(self._dim)]

    async def aembed(self, text: str) -> list[float]:
        return self.embed(text)


class FixedEmbeddings(Embeddings):
    """Returns pre-set vectors based on a lookup table."""

    __slots__ = ("_dim", "_vectors")

    def __init__(self, vectors: dict[str, list[float]], dim: int = 3) -> None:
        self._vectors = vectors
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return self._vectors.get(text, [0.0] * self._dim)

    async def aembed(self, text: str) -> list[float]:
        return self.embed(text)


# ---------------------------------------------------------------------------
# Embeddings ABC tests
# ---------------------------------------------------------------------------


class TestEmbeddingsABC:
    """Tests for the Embeddings abstract base class."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Embeddings()  # type: ignore[abstract]

    def test_mock_implements_abc(self) -> None:
        emb = MockEmbeddings(dim=8)
        assert isinstance(emb, Embeddings)
        assert emb.dimension == 8

    def test_mock_embed_sync(self) -> None:
        emb = MockEmbeddings(dim=3)
        vec = emb.embed("hello")
        assert len(vec) == 3
        assert all(isinstance(v, float) for v in vec)

    async def test_mock_embed_async(self) -> None:
        emb = MockEmbeddings(dim=3)
        vec = await emb.aembed("hello")
        assert len(vec) == 3

    def test_empty_text(self) -> None:
        emb = MockEmbeddings(dim=4)
        vec = emb.embed("")
        assert vec == [0.0, 0.0, 0.0, 0.0]


class TestOpenAIEmbeddings:
    """Tests for OpenAIEmbeddings (construction only — no real API calls)."""

    def test_is_embeddings_subclass(self) -> None:
        assert issubclass(OpenAIEmbeddings, Embeddings)

    def test_dimension_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch openai import
        import types

        mock_openai = types.ModuleType("openai")
        mock_openai.OpenAI = lambda **kw: None  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "openai", mock_openai)

        emb = OpenAIEmbeddings(model="test", dimension=768, api_key="fake")
        assert emb.dimension == 768


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for the _cosine_similarity helper."""

    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_known_value(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        dot = 1 * 4 + 2 * 5 + 3 * 6  # 32
        na = math.sqrt(14)
        nb = math.sqrt(77)
        expected = dot / (na * nb)
        assert _cosine_similarity(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# VectorMemoryStore — protocol conformance
# ---------------------------------------------------------------------------


class TestVectorProtocol:
    """VectorMemoryStore satisfies the MemoryStore protocol."""

    def test_isinstance_check(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        assert isinstance(store, MemoryStore)


# ---------------------------------------------------------------------------
# VectorMemoryStore — lifecycle
# ---------------------------------------------------------------------------


class TestVectorLifecycle:
    def test_init(self) -> None:
        emb = MockEmbeddings(dim=4)
        store = VectorMemoryStore(emb)
        assert store.embeddings is emb
        assert len(store) == 0

    def test_repr(self) -> None:
        store = VectorMemoryStore(MockEmbeddings(dim=8))
        assert "VectorMemoryStore" in repr(store)
        assert "dimension=8" in repr(store)


# ---------------------------------------------------------------------------
# VectorMemoryStore — add / get
# ---------------------------------------------------------------------------


class TestVectorAddGet:
    async def test_add_and_get(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        item = HumanMemory(content="hello world")
        await store.add(item)
        assert len(store) == 1
        got = await store.get(item.id)
        assert got is not None
        assert got.content == "hello world"

    async def test_get_nonexistent(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        assert await store.get("nope") is None

    async def test_add_computes_embedding(self) -> None:
        emb = MockEmbeddings(dim=3)
        store = VectorMemoryStore(emb)
        item = HumanMemory(content="test")
        await store.add(item)
        assert emb.call_count == 1
        assert item.id in store._vectors
        assert len(store._vectors[item.id]) == 3

    async def test_add_multiple(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        for i in range(5):
            await store.add(HumanMemory(content=f"item {i}"))
        assert len(store) == 5

    async def test_upsert(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        item = HumanMemory(id="x", content="v1")
        await store.add(item)
        updated = HumanMemory(id="x", content="v2")
        await store.add(updated)
        assert len(store) == 1
        got = await store.get("x")
        assert got is not None
        assert got.content == "v2"


# ---------------------------------------------------------------------------
# VectorMemoryStore — search (semantic)
# ---------------------------------------------------------------------------


class TestVectorSearch:
    async def test_semantic_ranking(self) -> None:
        """Items closer to query in embedding space rank higher."""
        vecs = {
            "cats are pets": [1.0, 0.0, 0.0],
            "dogs are pets": [0.9, 0.1, 0.0],
            "quantum physics": [0.0, 0.0, 1.0],
            "search for pets": [0.95, 0.05, 0.0],
        }
        emb = FixedEmbeddings(vecs, dim=3)
        store = VectorMemoryStore(emb)
        await store.add(HumanMemory(content="cats are pets"))
        await store.add(HumanMemory(content="dogs are pets"))
        await store.add(HumanMemory(content="quantum physics"))

        results = await store.search(query="search for pets", limit=3)
        contents = [r.content for r in results]
        # Pets-related should rank before quantum physics
        assert contents.index("cats are pets") < contents.index("quantum physics")
        assert contents.index("dogs are pets") < contents.index("quantum physics")

    async def test_search_no_query_returns_newest(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        item1 = HumanMemory(content="first", created_at="2024-01-01T00:00:00")
        item2 = HumanMemory(content="second", created_at="2024-01-02T00:00:00")
        await store.add(item1)
        await store.add(item2)
        results = await store.search()
        assert results[0].content == "second"

    async def test_search_limit(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        for i in range(10):
            await store.add(HumanMemory(content=f"item {i}"))
        results = await store.search(limit=3)
        assert len(results) == 3

    async def test_search_empty_store(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        results = await store.search(query="anything")
        assert results == []

    async def test_search_by_memory_type(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        await store.add(HumanMemory(content="user msg"))
        await store.add(SystemMemory(content="sys msg"))
        results = await store.search(memory_type="human")
        assert len(results) == 1
        assert results[0].memory_type == "human"

    async def test_search_by_status(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        item = HumanMemory(content="draft item", status=MemoryStatus.DRAFT)
        await store.add(item)
        await store.add(HumanMemory(content="accepted item"))
        results = await store.search(status=MemoryStatus.DRAFT)
        assert len(results) == 1
        assert results[0].content == "draft item"

    async def test_search_by_metadata(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        meta = MemoryMetadata(user_id="u1", session_id="s1")
        await store.add(HumanMemory(content="user1", metadata=meta))
        await store.add(HumanMemory(content="user2", metadata=MemoryMetadata(user_id="u2")))
        results = await store.search(metadata=MemoryMetadata(user_id="u1"))
        assert len(results) == 1
        assert results[0].content == "user1"

    async def test_search_combined_filters(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        meta = MemoryMetadata(user_id="u1")
        await store.add(HumanMemory(content="human u1", metadata=meta))
        await store.add(SystemMemory(content="system u1", metadata=meta))
        await store.add(HumanMemory(content="human u2", metadata=MemoryMetadata(user_id="u2")))
        results = await store.search(
            memory_type="human",
            metadata=MemoryMetadata(user_id="u1"),
        )
        assert len(results) == 1
        assert results[0].content == "human u1"


# ---------------------------------------------------------------------------
# VectorMemoryStore — clear
# ---------------------------------------------------------------------------


class TestVectorClear:
    async def test_clear_all(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        await store.add(HumanMemory(content="a"))
        await store.add(HumanMemory(content="b"))
        count = await store.clear()
        assert count == 2
        assert len(store) == 0

    async def test_clear_with_metadata(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        meta1 = MemoryMetadata(user_id="u1")
        meta2 = MemoryMetadata(user_id="u2")
        await store.add(HumanMemory(content="a", metadata=meta1))
        await store.add(HumanMemory(content="b", metadata=meta2))
        count = await store.clear(metadata=MemoryMetadata(user_id="u1"))
        assert count == 1
        assert len(store) == 1
        remaining = await store.search()
        assert remaining[0].content == "b"

    async def test_clear_removes_vectors(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        item = HumanMemory(content="test")
        await store.add(item)
        assert item.id in store._vectors
        await store.clear()
        assert item.id not in store._vectors

    async def test_clear_empty(self) -> None:
        store = VectorMemoryStore(MockEmbeddings())
        count = await store.clear()
        assert count == 0


# ---------------------------------------------------------------------------
# VectorMemoryStore — embedding call tracking
# ---------------------------------------------------------------------------


class TestVectorEmbeddingCalls:
    async def test_add_calls_embed(self) -> None:
        emb = MockEmbeddings()
        store = VectorMemoryStore(emb)
        await store.add(HumanMemory(content="a"))
        await store.add(HumanMemory(content="b"))
        assert emb.call_count == 2

    async def test_search_calls_embed_for_query(self) -> None:
        emb = MockEmbeddings()
        store = VectorMemoryStore(emb)
        await store.add(HumanMemory(content="stored"))
        emb.call_count = 0  # reset
        await store.search(query="find me")
        assert emb.call_count == 1  # only the query is embedded

    async def test_search_no_query_no_embed(self) -> None:
        emb = MockEmbeddings()
        store = VectorMemoryStore(emb)
        await store.add(HumanMemory(content="stored"))
        emb.call_count = 0
        await store.search()  # no query
        assert emb.call_count == 0
