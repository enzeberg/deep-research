"""Memory manager coordinating working and short-term memory.

``MemoryManager`` is the single entry point used by the workflow layer.  It
owns both memory tiers and exposes a snapshot API so the entire memory state
can be serialised to a plain dict (e.g. for checkpointing or debugging).
"""

from typing import Any

from src.memory.working import WorkingMemory
from src.memory.short_term import ShortTermMemory
from src.config import settings


class MemoryManager:
    """Facade over :class:`WorkingMemory` and :class:`ShortTermMemory`.

    Memory tiers
    ------------
    * **Working memory** (L1): bounded deque of in-flight items for the
      *current* session — plan, intermediate findings, etc.
    * **Short-term memory** (L2): sliding window of *completed* sessions,
      searchable via TF-IDF cosine similarity with recency decay.

    Snapshot / restore
    ------------------
    ``snapshot()`` returns a JSON-serialisable dict.  ``restore()`` rebuilds
    both tiers from that dict, enabling workflow checkpointing and test
    fixtures without touching any external storage.
    """

    def __init__(
        self,
        working_memory_size: int | None = None,
        short_term_memory_size: int | None = None,
    ):
        self.working_memory = WorkingMemory(
            max_size=working_memory_size or settings.max_working_memory_size
        )
        self.short_term_memory = ShortTermMemory(
            max_size=short_term_memory_size or settings.max_short_term_memory_size
        )

    # ── Write ─────────────────────────────────────────────────────────────

    def add_to_working(
        self, item_type: str, content: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        """Write an item into working memory (L1)."""
        self.working_memory.add(item_type, content, metadata)

    def save_session(
        self,
        query: str,
        plan: dict[str, Any],
        results: list[dict[str, Any]],
        report: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Commit a completed session to short-term memory (L2)."""
        self.short_term_memory.save_session(query, plan, results, report, metadata)

    # ── Read ──────────────────────────────────────────────────────────────

    def get_context(self) -> dict[str, Any]:
        """Return a combined view of both memory tiers for prompt injection."""
        return {
            "working_memory": self.working_memory.get_recent(),
            "short_term_memory": self.short_term_memory.get_recent_sessions(n=3),
            "working_summary": self.working_memory.get_context_summary(),
            "short_term_summary": self.short_term_memory.get_summary(),
        }

    def find_relevant_history(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return *top_k* past sessions most relevant to *query* (TF-IDF + recency)."""
        return self.short_term_memory.find_similar_queries(query, top_k)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def clear_working_memory(self) -> None:
        """Flush L1 (call at session boundaries)."""
        self.working_memory.clear()

    def reset(self) -> None:
        """Flush both tiers."""
        self.working_memory.clear()
        self.short_term_memory.clear()

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Serialise both tiers to a JSON-compatible dict."""
        return {
            "working_memory": self.working_memory.to_dict(),
            "short_term_memory": self.short_term_memory.to_dict(),
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore both tiers from a snapshot produced by :meth:`snapshot`."""
        self.working_memory = WorkingMemory.from_dict(data["working_memory"])
        self.short_term_memory = ShortTermMemory.from_dict(data["short_term_memory"])
