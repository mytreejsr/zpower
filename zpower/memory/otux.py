# zpower/memory/otux.py  —  OTUX-S v1.2.0
# v1.2 fixes:
#   CRITICAL: write_batch() result indexing bug fixed (buf_result_idx tracking)
#   PERF:     query() O(N log N) → O(N) via np.argpartition
#   FIX:      query_by_coord returns dicts (consistent with query())
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def _to_numpy(x) -> np.ndarray:
    if _TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _cosine_batch(matrix: np.ndarray, n: int, vec: np.ndarray) -> np.ndarray:
    active = matrix[:n]
    norms  = np.linalg.norm(active, axis=1) + 1e-10
    qnorm  = np.linalg.norm(vec) + 1e-10
    return (active @ vec) / (norms * qnorm)


@dataclass
class ImportanceWeights:
    novelty:     float = 0.35
    reward:      float = 0.30
    context_fit: float = 0.20
    recurrence:  float = 0.15

    def __post_init__(self):
        total = self.novelty + self.reward + self.context_fit + self.recurrence
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"ImportanceWeights must sum to 1.0 (got {total:.4f})")


@dataclass
class OtuxEntry:
    token:        str
    vector:       np.ndarray
    x:            float = 0.0
    y:            float = 0.0
    z:            float = 0.0
    metadata:     Dict  = field(default_factory=dict)
    reward:       float = 0.5
    timestamp:    float = field(default_factory=time.time)
    perf_score:   float = 0.0
    access_count: int   = 0


_GROWTH = 2


class OtuxStore:
    """
    OTUX-S v1.2.0 — Selective Context-Aware Memory Store.

    3D semantic coordinates:
      x = positional index / sequence position
      y = relational layer / domain category
      z = contextual depth / reasoning stage

    v1.1: O(1) amortized writes, write_batch(), fast hash
    v1.2: write_batch() result indexing fix (CRITICAL),
          query() O(N) via argpartition (was O(N log N)),
          query_by_coord() returns consistent dict format
    """

    def __init__(
        self,
        dim:                  int   = 256,
        mode:                 str   = "selective",
        importance_threshold: float = 0.65,
        forget_threshold:     float = 0.30,
        buffer_strikes:       int   = 3,
        weights:              Any   = None,
        context_window:       int   = 16,
        max_entries:          int   = 10_000,
        decay:                float = 0.95,
    ):
        self.dim          = dim
        self.mode         = mode
        self.theta_store  = importance_threshold
        self.theta_forget = forget_threshold
        self._strikes     = buffer_strikes
        self.max_entries  = max_entries
        self.decay        = decay

        if weights is None:
            self.w = ImportanceWeights()
        elif isinstance(weights, dict):
            self.w = ImportanceWeights(**weights)
        else:
            self.w = weights

        self._entries: List[OtuxEntry] = []
        self._cap:  int          = 64
        self._n:    int          = 0
        self._mat:  np.ndarray   = np.zeros((self._cap, dim), dtype=np.float32)
        self._context:    deque  = deque(maxlen=context_window)
        self._buffer:     Dict[str, int]   = {}
        self._buf_data:   Dict[str, tuple] = {}
        self._recurrence: Dict[str, int]   = {}
        self.stats = {"stored": 0, "buffered": 0, "discarded": 0, "evicted": 0}

    # ── Write single ───────────────────────────────────────────────────────

    def write(
        self,
        token:    str,
        vector:   Optional[Any] = None,
        x:        float = 0.0,
        y:        float = 0.0,
        z:        float = 0.0,
        metadata: Optional[Dict] = None,
        reward:   float = 0.5,
        force:    bool  = False,
    ) -> str:
        vec = self._prep(vector)
        h   = self._hash(token)
        self._recurrence[h] = self._recurrence.get(h, 0) + 1
        md  = metadata or {}

        if force or self.mode == "full":
            self._do_store(token, vec, x, y, z, md, reward, 1.0)
            return "forced" if force else "stored"

        score = self._importance(vec, reward, h)

        if score >= self.theta_store:
            self._do_store(token, vec, x, y, z, md, reward, score)
            self.stats["stored"] += 1
            return "stored"

        if score < self.theta_forget:
            self.stats["discarded"] += 1
            return "discarded"

        self._buffer[h]   = self._buffer.get(h, 0) + 1
        self._buf_data[h] = (token, vec, x, y, z, md, reward, score)
        self.stats["buffered"] += 1

        if self._buffer[h] >= self._strikes:
            args = self._buf_data.pop(h)
            del self._buffer[h]
            self._do_store(*args[:1], args[1], *args[2:7], args[7])
            self.stats["stored"] += 1
            return "stored"
        return "buffered"

    # ── Write batch — v1.2: CRITICAL bug fix ──────────────────────────────

    def write_batch(self, entries: List[Dict]) -> List[str]:
        """
        Write multiple entries with a single matrix update.

        v1.2 fix: buf_result_idx tracks which results[] index corresponds
        to each buffered token. On promotion, updates the CORRECT index
        instead of blindly overwriting results[-1].
        """
        results:       List[str]       = []
        to_store:      List[tuple]     = []
        buf_result_idx: Dict[str, int] = {}   # hash → index in results[]

        for e in entries:
            token  = e["token"]
            vec    = self._prep(e.get("vector"))
            x      = float(e.get("x", 0.0))
            y      = float(e.get("y", 0.0))
            z      = float(e.get("z", 0.0))
            md     = e.get("metadata") or {}
            reward = float(e.get("reward", 0.5))
            force  = bool(e.get("force", False))
            h      = self._hash(token)
            self._recurrence[h] = self._recurrence.get(h, 0) + 1
            idx    = len(results)   # current position in results list

            if force or self.mode == "full":
                to_store.append((token, vec, x, y, z, md, reward, 1.0))
                results.append("forced" if force else "stored")
                continue

            score = self._importance_with_pending(vec, reward, h, to_store)

            if score >= self.theta_store:
                to_store.append((token, vec, x, y, z, md, reward, score))
                self.stats["stored"] += 1
                results.append("stored")

            elif score < self.theta_forget:
                self.stats["discarded"] += 1
                results.append("discarded")

            else:
                self._buffer[h]   = self._buffer.get(h, 0) + 1
                self._buf_data[h] = (token, vec, x, y, z, md, reward, score)
                self.stats["buffered"] += 1
                buf_result_idx[h] = idx   # remember THIS entry's position
                results.append("buffered")

                if self._buffer[h] >= self._strikes:
                    args = self._buf_data.pop(h)
                    del self._buffer[h]
                    to_store.append(args)
                    self.stats["stored"] += 1
                    # v1.2 fix: update correct index, not results[-1]
                    results[buf_result_idx.pop(h)] = "stored"

        if to_store:
            self._bulk_insert(to_store)

        return results

    # ── Query — v1.2: O(N) via argpartition ───────────────────────────────

    def query(
        self,
        vector:    Any,
        top_k:     int   = 10,
        threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Cosine-similarity retrieval.
        v1.2: O(N) argpartition replaces O(N log N) argsort.
        """
        if self._n == 0:
            return []

        vec  = self._prep(vector)
        sims = _cosine_batch(self._mat, self._n, vec)

        k = min(top_k, self._n)
        # O(N) partial selection — only sort the top-k among themselves
        top_idx_unsorted = np.argpartition(sims, -k)[-k:]
        top_idx          = top_idx_unsorted[np.argsort(sims[top_idx_unsorted])[::-1]]

        results = []
        for i in top_idx:
            if float(sims[i]) < threshold:
                continue
            e = self._entries[i]
            e.access_count += 1
            results.append({
                "token":      e.token,
                "vector":     e.vector.copy(),
                "x": e.x, "y": e.y, "z": e.z,
                "metadata":   e.metadata,
                "sim":        float(sims[i]),
                "reward":     e.reward,
                "perf_score": e.perf_score,
            })

        self._context.append(vec.copy())
        return results

    # ── Query by coord — v1.2: consistent dict return ─────────────────────

    def query_by_coord(
        self,
        x:   Optional[float] = None,
        y:   Optional[float] = None,
        z:   Optional[float] = None,
        tol: float = 0.5,
    ) -> List[Dict]:
        """
        Retrieve entries near a semantic coordinate.
        v1.2: returns dicts (same format as query()) instead of OtuxEntry objects.
        """
        out = []
        for e in self._entries:
            if x is not None and abs(e.x - x) > tol: continue
            if y is not None and abs(e.y - y) > tol: continue
            if z is not None and abs(e.z - z) > tol: continue
            out.append({
                "token":      e.token,
                "vector":     e.vector.copy(),
                "x": e.x, "y": e.y, "z": e.z,
                "metadata":   e.metadata,
                "sim":        1.0,
                "reward":     e.reward,
                "perf_score": e.perf_score,
            })
        return out

    # ── Stats ──────────────────────────────────────────────────────────────

    def filter_stats(self) -> Dict:
        total = sum(self.stats.values())
        cr = 0.0
        if total > 0:
            cr = round(100.0 * (1.0 - self.stats["stored"] / max(total, 1)), 1)
        return {
            **self.stats,
            "total_seen":        total,
            "currently_stored":  self._n,
            "matrix_capacity":   self._cap,
            "compression_ratio": f"{cr}%",
        }

    def clear(self):
        self._entries.clear()
        self._n = 0; self._cap = 64
        self._mat = np.zeros((self._cap, self.dim), dtype=np.float32)
        self._context.clear(); self._buffer.clear()
        self._buf_data.clear(); self._recurrence.clear()
        self.stats = {"stored": 0, "buffered": 0, "discarded": 0, "evicted": 0}

    def health(self) -> Dict:
        return {"status": "ok", **self.filter_stats()}

    def status(self) -> Dict:
        return self.filter_stats()

    def __len__(self):   return self._n
    def __repr__(self):
        s = self.filter_stats()
        return (f"OtuxStore(dim={self.dim}, mode={self.mode}, "
                f"stored={self._n}/{self._cap}, compression={s['compression_ratio']})")

    # ── Internal ───────────────────────────────────────────────────────────

    def _do_store(self, token, vec, x, y, z, md, reward, score):
        entry = OtuxEntry(token=token, vector=vec.copy(), x=x, y=y, z=z,
                          metadata=md, reward=reward, perf_score=score)
        self._entries.append(entry)
        self._append_to_matrix(vec)
        self._context.append(vec.copy())
        if self._n > self.max_entries:
            self._evict()

    def _bulk_insert(self, items: List[Tuple]):
        needed = self._n + len(items)
        if needed > self._cap:
            new_cap = max(needed * _GROWTH, self._cap * _GROWTH)
            new_mat = np.zeros((new_cap, self.dim), dtype=np.float32)
            if self._n > 0:
                new_mat[:self._n] = self._mat[:self._n]
            self._mat = new_mat
            self._cap = new_cap
        for token, vec, x, y, z, md, reward, score in items:
            entry = OtuxEntry(token=token, vector=vec.copy(), x=x, y=y, z=z,
                              metadata=md, reward=reward, perf_score=score)
            self._entries.append(entry)
            self._mat[self._n] = vec
            self._n += 1
            self._context.append(vec.copy())
        while self._n > self.max_entries:
            self._evict()

    def _append_to_matrix(self, vec: np.ndarray):
        if self._n >= self._cap:
            new_cap = self._cap * _GROWTH
            new_mat = np.zeros((new_cap, self.dim), dtype=np.float32)
            new_mat[:self._n] = self._mat[:self._n]
            self._mat = new_mat
            self._cap = new_cap
        self._mat[self._n] = vec
        self._n += 1

    def _evict(self):
        if self._n == 0: return
        scores = []
        for e in self._entries:
            d = self.decay if e.reward <= 0.8 else (self.decay + 1.0) / 2.0
            scores.append(e.perf_score * (d ** e.access_count))
        worst = int(np.argmin(scores))
        last  = self._n - 1
        if worst != last:
            self._mat[worst] = self._mat[last]
            self._entries[worst] = self._entries[last]
        self._mat[last] = 0.0
        self._entries.pop(last)
        self._n -= 1
        self.stats["evicted"] += 1

    def _importance(self, vec, reward, h):
        return self._importance_with_pending(vec, reward, h, [])

    def _importance_with_pending(self, vec, reward, h, pending):
        if self._n > 0:
            sims    = _cosine_batch(self._mat, self._n, vec)
            novelty = float(1.0 - np.max(np.clip(sims, 0, 1)))
        else:
            novelty = 1.0
        for pv_args in pending:
            pv = pv_args[1] if isinstance(pv_args, (list, tuple)) else pv_args
            psim = float(np.dot(vec, pv) /
                         (np.linalg.norm(vec) * np.linalg.norm(pv) + 1e-10))
            novelty = max(0.0, min(novelty, 1.0 - psim))
        if self._context:
            ctx_mat = np.stack(list(self._context))
            cf      = _cosine_batch(ctx_mat, len(self._context), vec)
            ctx_fit = float(np.max(np.clip(cf, 0, 1)))
        else:
            ctx_fit = 0.5
        count      = self._recurrence.get(h, 1)
        recurrence = min(math.log1p(count) / 5.0, 1.0)
        rew   = float(np.clip(reward, 0.0, 1.0))
        score = (self.w.novelty * novelty + self.w.reward * rew
               + self.w.context_fit * ctx_fit + self.w.recurrence * recurrence)
        return float(np.clip(score, 0.0, 1.0))

    def _prep(self, vec: Any) -> np.ndarray:
        if vec is None:
            return np.zeros(self.dim, dtype=np.float32)
        v = _to_numpy(vec).flatten()
        if len(v) != self.dim:
            raise ValueError(f"OtuxStore: vector dim mismatch (expected {self.dim}, got {len(v)})")
        norm = np.linalg.norm(v)
        return (v / (norm + 1e-10)).astype(np.float32)

    def _hash(self, token: str) -> str:
        return str(abs(hash(token)) % (10**15))
