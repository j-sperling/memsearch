"""Cloudflare Agent Memory backend (stub).

Scheme-gated alternative to :class:`~memsearch.store.MilvusStore`,
selected by URIs of the form ``agentmem://<account>/<namespace>``.

This module is a scheme placeholder: it lets the factory wiring in
``memsearch.core`` dispatch on ``agentmem://`` without requiring a
network client. All store methods raise :class:`NotImplementedError`
until the Cloudflare Agent Memory HTTP surface is actually wired.

Upstream API reference:
    https://developers.cloudflare.com/agents/api-reference/agents-api/

TODO — implement against the Cloudflare Agent Memory HTTP API:
    * ``put``        -- POST memory records keyed by ``chunk_hash``
                        (maps to :meth:`AgentMemoryStore.upsert`).
    * ``get``        -- GET a record by key
                        (maps to :meth:`AgentMemoryStore.query` with a
                        ``chunk_hash == "..."`` filter).
    * ``list``       -- paginated listing, filtered by ``source``
                        (maps to :meth:`AgentMemoryStore.indexed_sources`,
                        :meth:`AgentMemoryStore.hashes_by_source`,
                        :meth:`AgentMemoryStore.query`).
    * ``search``     -- vector + keyword query endpoint
                        (maps to :meth:`AgentMemoryStore.search`).
    * ``delete``     -- by key or by ``source`` filter
                        (maps to :meth:`AgentMemoryStore.delete_by_hashes`
                        and :meth:`AgentMemoryStore.delete_by_source`).

Deliberately no HTTP client is imported yet; this unit is
scheme-gating + stub only. A follow-up will add the transport.
"""

from __future__ import annotations

from typing import Any

from .store import Store

_NOT_WIRED = "Cloudflare Agent Memory backend not wired yet; see TODO in agent_memory_store.py"


class AgentMemoryStore(Store):
    """Stub backend selected by ``agentmem://`` URIs.

    Mirrors :class:`~memsearch.store.MilvusStore`'s constructor shape so
    the factory can swap implementations transparently. All operations
    currently raise :class:`NotImplementedError`.
    """

    def __init__(
        self,
        uri: str = "agentmem://",
        *,
        token: str | None = None,
        collection: str = "memsearch_chunks",
        dimension: int | None = 1536,
        description: str = "",
        **_ignored: Any,
    ) -> None:
        self._uri = uri
        self._token = token
        self._collection = collection
        self._dimension = dimension
        self._description = description

    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        raise NotImplementedError(_NOT_WIRED)

    def search(
        self,
        query_embedding: list[float],
        *,
        query_text: str = "",
        top_k: int = 10,
        filter_expr: str = "",
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(_NOT_WIRED)

    def query(self, *, filter_expr: str = "") -> list[dict[str, Any]]:
        raise NotImplementedError(_NOT_WIRED)

    def hashes_by_source(self, source: str) -> set[str]:
        raise NotImplementedError(_NOT_WIRED)

    def indexed_sources(self) -> set[str]:
        raise NotImplementedError(_NOT_WIRED)

    def delete_by_source(self, source: str) -> None:
        raise NotImplementedError(_NOT_WIRED)

    def delete_by_hashes(self, hashes: list[str]) -> None:
        raise NotImplementedError(_NOT_WIRED)

    def count(self) -> int:
        raise NotImplementedError(_NOT_WIRED)

    def drop(self) -> None:
        raise NotImplementedError(_NOT_WIRED)

    def close(self) -> None:
        # No resources held by the stub; safe no-op so ``with`` blocks
        # and ``MemSearch.close()`` don't explode during tests.
        return None

    def __enter__(self) -> AgentMemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
