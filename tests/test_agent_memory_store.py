"""Tests for the scheme-gated store factory and AgentMemoryStore stub.

Covers three things:
  * ``agentmem://`` URIs route to :class:`AgentMemoryStore`.
  * Non-``agentmem://`` URIs stay on :class:`MilvusStore` (Milvus mocked
    to avoid spinning up Milvus Lite for this unit test).
  * Every stub method raises :class:`NotImplementedError` with a
    pointer back to the TODO.
"""

from __future__ import annotations

from typing import Any

import pytest

from memsearch import core as core_module
from memsearch.agent_memory_store import AgentMemoryStore
from memsearch.core import MemSearch
from memsearch.store import MilvusStore, Store


class _FakeEmbedder:
    model_name = "fake-embed"
    dimension = 4
    batch_size = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> _FakeEmbedder:
    """Swap get_provider for a zero-cost fake so MemSearch() is cheap."""
    fake = _FakeEmbedder()
    monkeypatch.setattr(core_module, "get_provider", lambda *_a, **_kw: fake)
    return fake


def test_agentmem_scheme_routes_to_agent_memory_store(fake_embedder: _FakeEmbedder) -> None:
    mem = MemSearch(milvus_uri="agentmem://acct/ns")
    try:
        assert isinstance(mem.store, AgentMemoryStore)
        # Structural conformance to the Store protocol.
        assert isinstance(mem.store, Store)
    finally:
        mem.close()


def test_non_agentmem_scheme_routes_to_milvus_store(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedder: _FakeEmbedder,
) -> None:
    """Local path / http(s) / tcp / milvus:// -> MilvusStore.

    Stubs MilvusStore.__init__ so we don't need a live Milvus instance
    just to confirm the factory picked the right class.
    """
    captured: dict[str, Any] = {}

    def _fake_init(self: MilvusStore, **kwargs: Any) -> None:
        captured.update(kwargs)
        # Minimal attrs so MemSearch.close() works.
        self._client = None
        self._is_lite = False
        self._resolved_uri = kwargs.get("uri", "")
        self._collection = kwargs.get("collection", "")
        self._dimension = kwargs.get("dimension")

    def _fake_close(self: MilvusStore) -> None:
        return None

    monkeypatch.setattr(MilvusStore, "__init__", _fake_init)
    monkeypatch.setattr(MilvusStore, "close", _fake_close)

    for uri in ("/tmp/test.db", "http://localhost:19530", "milvus://localhost:19530"):
        mem = MemSearch(milvus_uri=uri)
        try:
            assert isinstance(mem.store, MilvusStore)
            assert not isinstance(mem.store, AgentMemoryStore)
            assert captured["uri"] == uri
        finally:
            mem.close()


def test_agent_memory_store_methods_raise_not_implemented() -> None:
    s = AgentMemoryStore(uri="agentmem://acct/ns", dimension=4)

    expected_msg = "Cloudflare Agent Memory backend not wired yet"

    with pytest.raises(NotImplementedError, match=expected_msg):
        s.upsert([{"chunk_hash": "h"}])
    with pytest.raises(NotImplementedError, match=expected_msg):
        s.search([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(NotImplementedError, match=expected_msg):
        s.query()
    with pytest.raises(NotImplementedError, match=expected_msg):
        s.hashes_by_source("a.md")
    with pytest.raises(NotImplementedError, match=expected_msg):
        s.indexed_sources()
    with pytest.raises(NotImplementedError, match=expected_msg):
        s.delete_by_source("a.md")
    with pytest.raises(NotImplementedError, match=expected_msg):
        s.delete_by_hashes(["h"])
    with pytest.raises(NotImplementedError, match=expected_msg):
        s.count()
    with pytest.raises(NotImplementedError, match=expected_msg):
        s.drop()

    # close() is a deliberate no-op so context-manager exit doesn't blow up.
    s.close()


def test_agent_memory_store_accepts_milvus_constructor_shape() -> None:
    """Same required + optional params as MilvusStore; extras are ignored."""
    s = AgentMemoryStore(
        uri="agentmem://acct/ns",
        token="tok",
        collection="memsearch_chunks",
        dimension=1536,
        description="desc",
        future_param="ignored",
    )
    assert isinstance(s, Store)
