"""Short-term memory for recent research history.

Retrieval is based on **TF-IDF cosine similarity** with a **recency decay**
penalty, giving recent sessions a boost over older but lexically similar ones.
Everything is implemented in pure Python (math + collections) — no extra deps.

Similarity formula
------------------
    final_score = cosine_sim(query_tfidf, session_tfidf) * recency_factor

    recency_factor = exp(-λ * age_in_hours)   λ = RECENCY_DECAY_RATE
"""

from __future__ import annotations

import math
from collections import deque, Counter
from datetime import datetime
from typing import Any


RECENCY_DECAY_RATE = 0.02  # λ — halves relevance after ~35 hours


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace/punctuation, drop short tokens."""
    import re
    return [t for t in re.split(r"[\s\W]+", text.lower()) if len(t) > 1]


def _tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency: count / total."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    common = vec_a.keys() & vec_b.keys()
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ShortTermMemory:
    """Sliding-window store for completed research sessions.

    Design notes
    ------------
    * Storage     : ``deque(maxlen=max_size)`` — O(1) append/evict (FIFO).
    * IDF corpus  : rebuilt lazily on each ``find_similar_queries`` call.
      With n ≤ 20 sessions this is fast enough; a real system would maintain
      an incremental IDF index.
    * Retrieval   : TF-IDF cosine similarity + exponential recency decay.
      This surfaces results that are both *topically* and *temporally* close
      to the current query — better than pure Jaccard for multi-word queries.
    * Snapshot    : ``to_dict`` / ``from_dict`` for JSON-serialisable state.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._sessions: deque[dict[str, Any]] = deque(maxlen=max_size)

    # ── Write ─────────────────────────────────────────────────────────────

    def save_session(
        self,
        query: str,
        plan: dict[str, Any],
        results: list[dict[str, Any]],
        report: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a completed research session."""
        self._sessions.append(
            {
                "query": query,
                "plan": plan,
                "results": results,
                "report": report,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
            }
        )

    def clear(self) -> None:
        """Discard all sessions."""
        self._sessions.clear()

    # ── Read ──────────────────────────────────────────────────────────────

    def get_recent_sessions(self, n: int | None = None) -> list[dict[str, Any]]:
        """Return the *n* most-recent sessions (all if *n* is None)."""
        sessions = list(self._sessions)
        return sessions if n is None else sessions[-n:]

    def find_similar_queries(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the *top_k* most relevant past sessions for *query*.

        Scoring uses TF-IDF cosine similarity weighted by a recency decay so
        that recent sessions are preferred when lexical similarity is equal.
        Falls back to zero-score entries only when the corpus is empty.
        """
        if not self._sessions:
            return []

        # Build IDF from the corpus (all stored queries)
        corpus_tokens = [_tokenize(s["query"]) for s in self._sessions]
        doc_count = len(corpus_tokens)
        df: Counter[str] = Counter()
        for tokens in corpus_tokens:
            df.update(set(tokens))
        idf: dict[str, float] = {
            term: math.log((doc_count + 1) / (freq + 1)) + 1
            for term, freq in df.items()
        }

        def tfidf(tokens: list[str]) -> dict[str, float]:
            tf = _tf(tokens)
            return {t: tf[t] * idf.get(t, 1.0) for t in tf}

        query_vec = tfidf(_tokenize(query))
        now = datetime.now()

        scored: list[tuple[float, dict[str, Any]]] = []
        for session, tokens in zip(self._sessions, corpus_tokens):
            sim = _cosine(query_vec, tfidf(tokens))

            # Recency decay: penalise older sessions
            age_hours = (now - datetime.fromisoformat(session["timestamp"])).total_seconds() / 3600
            recency = math.exp(-RECENCY_DECAY_RATE * age_hours)

            scored.append((sim * recency, session))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def get_summary(self) -> str:
        """Return a human-readable summary of stored sessions."""
        if not self._sessions:
            return "No sessions in short-term memory."

        lines = [f"Short-term Memory ({len(self._sessions)} sessions):"]
        for i, session in enumerate(self._sessions, 1):
            lines.append(f"{i}. {session['query'][:80]}  ({session['timestamp']})")
        return "\n".join(lines)

    # ── Snapshot ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise session store to a JSON-compatible dict."""
        return {"max_size": self.max_size, "sessions": list(self._sessions)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShortTermMemory":
        """Restore a ``ShortTermMemory`` instance from a snapshot dict."""
        instance = cls(max_size=data["max_size"])
        for session in data.get("sessions", []):
            instance._sessions.append(session)
        return instance

    # ── Dunder ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._sessions)
