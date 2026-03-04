# Memory System

This document describes the design, core algorithms, and engineering trade-offs
of the `src/memory/` module.

---

## 1. Why Memory?

Large language models are **stateless** — each call is independent.
Deep-Research needs to maintain context across two dimensions:

| Dimension | Problem | Solution |
|-----------|---------|----------|
| **Within a session** | Plan → Findings must be mutually aware | WorkingMemory (L1) |
| **Across sessions** | Repeated queries on similar topics waste work | ShortTermMemory (L2) |

---

## 2. Two-Tier Architecture

```
┌──────────────────────────────────────────────┐
│              MemoryManager (Facade)           │
│                                              │
│   ┌─────────────────┐  ┌──────────────────┐  │
│   │  WorkingMemory  │  │ ShortTermMemory  │  │
│   │      (L1)       │  │      (L2)        │  │
│   │                 │  │                  │  │
│   │  deque(maxlen)  │  │  deque(maxlen)   │  │
│   │  priority sort  │  │  TF-IDF + decay  │  │
│   └─────────────────┘  └──────────────────┘  │
│                                              │
│   snapshot() ←──── to_dict / from_dict ────▶ │
└──────────────────────────────────────────────┘
```

### 2.1 WorkingMemory (L1)

**Responsibility**: Hold intermediate state — plan, findings, etc. — for the
duration of a single Research Workflow execution, making it available to
downstream nodes.

**Core data structure**: `collections.deque(maxlen=max_size)`

- **Write**: O(1); oldest entry is evicted automatically when `maxlen` is
  reached (FIFO eviction)
- **Read**: O(n), n ≤ `max_size` (default 5) — negligible
- **Priority**: each `item_type` carries a predefined weight
  (`plan=10 > findings=8 > query=6 > result=4`) that influences the ordering
  in `get_context_summary()` but **never affects eviction order** — a
  deliberate trade-off to keep LRU semantics simple and predictable

```python
_TYPE_PRIORITY = {"plan": 10, "findings": 8, "query": 6, "result": 4}
```

### 2.2 ShortTermMemory (L2)

**Responsibility**: Persist completed research sessions and support semantic
similarity search so the Planner can incorporate relevant historical context.

**Core data structure**: `deque(maxlen=max_size)` (default 10 sessions), FIFO
eviction of the oldest session when full.

---

## 3. Retrieval Algorithm

### 3.1 Why Not Jaccard?

A naive set-intersection approach (Jaccard similarity) is sensitive to
high-frequency stop words and has poor discriminative power for multi-word
queries:

```
sim(A, B) = |A ∩ B| / |A ∪ B|
```

### 3.2 TF-IDF Cosine Similarity

Each query string is represented as a **TF-IDF vector**; retrieval ranks
sessions by cosine similarity against the incoming query:

```
TF(t, d)    = count(t in d) / len(d)

IDF(t)      = log((N+1) / (df(t)+1)) + 1    # +1 smoothing avoids log(0)

TFIDF(t, d) = TF(t, d) × IDF(t)

cosine(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

**Benefits**:
- Terms that appear in every document receive a low IDF and are down-weighted
  automatically
- Length normalisation via the cosine denominator removes the effect of query
  length differences
- Pure Python (`math` + `collections`) — no additional dependencies required

### 3.3 Recency Decay

Lexical similarity alone can surface stale sessions. An **exponential decay**
factor penalises older entries:

```
recency_factor = exp(−λ × age_in_hours)     λ = 0.02

final_score = cosine_sim × recency_factor
```

Decay curve:

```
recency_factor
    1.0 │●
        │  ●
    0.5 │     ●          ← ~35 hours → 0.5
        │          ●
    0.1 │               ●●
        └─────────────────────── age (hours)
          0   10   20   35   80
```

With `λ = 0.02`, relevance halves after ~35 hours — appropriate for a
"revisit the next day" research pattern. Increasing λ biases results toward
more recent sessions; decreasing it makes the system more conservative.

---

## 4. Snapshot / Restore

Both `WorkingMemory` and `ShortTermMemory` implement `to_dict()` /
`from_dict()`. `MemoryManager` exposes a unified `snapshot()` / `restore()`
API:

```python
# Serialise
state = memory_manager.snapshot()
# → {"working_memory": {...}, "short_term_memory": {...}}

# Deserialise
memory_manager.restore(state)
```

**Use cases**:
- **Workflow checkpointing**: a failed LangGraph node can resume from the last
  snapshot instead of re-executing the entire pipeline
- **Test fixtures**: unit tests can inject a pre-built memory state directly,
  without mocking the full workflow
- **Debugging**: write the snapshot to a JSON file for offline analysis of a
  failed or unexpected session

The output is a plain Python dict that is directly `json.dumps()`-serialisable
— no `pickle`, avoiding version-compatibility issues.

---

## 5. Integration with the Workflow

```
Plan Node
  ├── memory_manager.get_context()                  # inject history into prompt
  └── memory_manager.add_to_working("plan", plan)   # write to L1

Research Node
  └── memory_manager.add_to_working("findings", ...)  # write to L1

Report Node
  └── memory_manager.save_session(...)              # promote to L2
```

`MemoryManager.get_context()` returns four fields for Planner prompt injection:

| Field | Source | Purpose |
|-------|--------|---------|
| `working_memory` | L1 raw items | debugging |
| `short_term_memory` | L2 most-recent 3 sessions | debugging |
| `working_summary` | L1 priority-sorted text | prompt injection |
| `short_term_summary` | L2 session list | prompt injection |

---

## 6. Design Trade-offs and Limitations

| Decision | Choice | Rejected alternative | Rationale |
|----------|--------|----------------------|-----------|
| Storage | In-process deque | Redis / SQLite | No extra infra; restart-to-clear fits "short-term" semantics |
| Retrieval | TF-IDF (pure Python) | Embedding vectors + vector DB | No GPU/API call needed; n ≤ 20 sessions is sufficient |
| Eviction | FIFO | LRU / LFU | Recency matters more than frequency in research; FIFO is predictable |
| IDF update | Full rebuild per query | Incremental maintenance | Full rebuild < 1 ms at n ≤ 20; incremental adds complexity without benefit |

### Known Limitations

- **Not persistent across restarts**: L2 is lost when the process exits.
  Persistence can be added by writing `snapshot()` output to disk and calling
  `restore()` at startup.
- **IDF quality degrades with few sessions**: with fewer than ~3 documents,
  IDF scores are not meaningful and Jaccard may actually be more stable.
- **No semantic understanding**: TF-IDF is a bag-of-words model; "machine
  learning" and "ML" are not recognised as synonymous. Embedding-based
  retrieval would address this at larger scale.

---

## 7. Potential Extensions

The interfaces are designed so the following extensions can be added without
breaking existing callers:

1. **Persistence layer**: add a `storage` parameter to `MemoryManager` that
   accepts a `FileStorage` / `RedisStorage` strategy; `snapshot()` / `restore()`
   delegate to it automatically
2. **Embedding retrieval**: swap the internals of
   `ShortTermMemory.find_similar_queries()` for a vector similarity search;
   the public interface stays the same
3. **Long-term memory (L3)**: add a `LongTermMemory` class backed by a vector
   database for cross-user knowledge; surface it through the existing
   `find_relevant_history()` call on `MemoryManager`
4. **TTL expiry**: record a TTL in `save_session()` and filter expired entries
   in `get_recent_sessions()`
