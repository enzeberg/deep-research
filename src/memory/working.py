"""Working memory for current session context.

Implements a bounded, priority-aware buffer for in-flight research data.
Items are evicted by age (FIFO via deque) but can be retrieved by type or
recency. A lightweight snapshot mechanism allows the caller to save/restore
the full buffer state (useful for workflow checkpointing).
"""

from typing import Any
from collections import deque
from datetime import datetime

# Higher number = shown first in context summaries.
_TYPE_PRIORITY: dict[str, int] = {
    "plan": 10,
    "findings": 8,
    "query": 6,
    "result": 4,
}
_DEFAULT_PRIORITY = 2


class WorkingMemory:
    """Bounded working-memory buffer for a single research session.

    Design notes
    ------------
    * Storage  : ``collections.deque`` with ``maxlen`` — O(1) append/evict.
    * Retrieval: O(n) scan; n ≤ max_size (default 5) so this is negligible.
    * Priority : only affects *summary ordering*, never eviction order.  Eviction
      stays FIFO so the LRU invariant is easy to reason about.
    * Snapshot : ``to_dict`` / ``from_dict`` give a plain-Python snapshot that is
      JSON-serialisable, allowing workflow checkpointing without extra deps.
    """

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._memory: deque[dict[str, Any]] = deque(maxlen=max_size)

    # ── Write ─────────────────────────────────────────────────────────────

    def add(self, item_type: str, content: Any, metadata: dict[str, Any] | None = None) -> None:
        """Append an item to the buffer, evicting the oldest entry if full."""
        self._memory.append(
            {
                "type": item_type,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "priority": _TYPE_PRIORITY.get(item_type, _DEFAULT_PRIORITY),
            }
        )

    def clear(self) -> None:
        """Discard all items."""
        self._memory.clear()

    # ── Read ──────────────────────────────────────────────────────────────

    def get_recent(self, n: int | None = None) -> list[dict[str, Any]]:
        """Return the *n* most-recently added items (all if *n* is None)."""
        items = list(self._memory)
        return items if n is None else items[-n:]

    def get_by_type(self, item_type: str) -> list[dict[str, Any]]:
        """Return all items whose ``type`` matches *item_type*."""
        return [item for item in self._memory if item["type"] == item_type]

    def get_context_summary(self) -> str:
        """Return a human-readable summary ordered by item priority (desc)."""
        if not self._memory:
            return "No items in working memory."

        ordered = sorted(self._memory, key=lambda x: x["priority"], reverse=True)
        lines = ["Working Memory Context:"]
        for item in ordered:
            lines.append(f"- [{item['type']}] {str(item['content'])[:100]}")
        return "\n".join(lines)

    # ── Snapshot ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise buffer state to a JSON-compatible dict."""
        return {"max_size": self.max_size, "items": list(self._memory)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkingMemory":
        """Restore a ``WorkingMemory`` instance from a snapshot dict."""
        instance = cls(max_size=data["max_size"])
        for item in data.get("items", []):
            instance._memory.append(item)
        return instance

    # ── Dunder ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._memory)
