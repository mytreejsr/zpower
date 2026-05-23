# tests/test_otux.py
import numpy as np
import pytest
from zpower.memory.otux import OtuxStore, ImportanceWeights


def make_vec(dim=256, seed=None):
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


# ── ImportanceWeights ──────────────────────────────────────────────────────

def test_weights_sum_to_one():
    w = ImportanceWeights(0.35, 0.30, 0.20, 0.15)
    assert abs(w.novelty + w.reward + w.context_fit + w.recurrence - 1.0) < 1e-5


def test_weights_bad_sum():
    with pytest.raises(ValueError):
        ImportanceWeights(0.5, 0.5, 0.5, 0.5)


def test_weights_repr():
    w = ImportanceWeights()
    assert "ImportanceWeights" in repr(w)


# ── OtuxStore basic ────────────────────────────────────────────────────────

def test_store_and_len():
    store = OtuxStore(dim=8, mode="full")
    store.write("hello", make_vec(8, 0))
    store.write("world", make_vec(8, 1))
    assert len(store) == 2


def test_query_returns_results():
    store = OtuxStore(dim=8, mode="full")
    v = make_vec(8, 42)
    store.write("alpha", v)
    results = store.query(v, top_k=1)
    assert len(results) == 1
    assert results[0]["token"] == "alpha"
    assert results[0]["sim"] > 0.99


def test_query_empty_store():
    store = OtuxStore(dim=8)
    results = store.query(make_vec(8), top_k=5)
    assert results == []


def test_coord_retrieval():
    store = OtuxStore(dim=8, mode="full")
    store.write("deep",    make_vec(8, 0), x=0, y=0, z=3)
    store.write("surface", make_vec(8, 1), x=0, y=0, z=0)
    deep_results = store.query_by_coord(z=3, tol=0.1)
    assert len(deep_results) == 1
    # v1.3 FIX: query_by_coord returns dicts, not OtuxEntry objects
    assert deep_results[0]["token"] == "deep"


def test_force_write_bypasses_gate():
    store = OtuxStore(dim=8, mode="selective", importance_threshold=0.99)
    result = store.write("forced", make_vec(8, 0), reward=0.0, force=True)
    assert result == "forced"
    assert len(store) == 1


# ── Importance gate ────────────────────────────────────────────────────────

def test_discard_low_reward():
    store = OtuxStore(
        dim=8,
        mode="selective",
        importance_threshold=0.65,
        forget_threshold=0.30,
        weights={"novelty": 0.35, "reward": 0.30, "context_fit": 0.20, "recurrence": 0.15},
    )
    result = store.write("junk", make_vec(8, 99), reward=0.0)
    assert result in ("discarded", "buffered")


def test_high_reward_stored():
    store = OtuxStore(dim=8, mode="selective", importance_threshold=0.65)
    result = store.write("important", make_vec(8, 1), reward=1.0)
    assert result == "stored"


def test_buffer_three_strikes():
    store = OtuxStore(
        dim=8, mode="selective",
        importance_threshold=0.90,
        forget_threshold=0.05,
        buffer_strikes=3,
    )
    v = make_vec(8, 7)
    for _ in range(3):
        store.write("repeat", v, reward=0.4)
    assert len(store) >= 1


# ── Filter stats ───────────────────────────────────────────────────────────

def test_filter_stats_keys():
    store = OtuxStore(dim=8, mode="full")
    store.write("a", make_vec(8, 0))
    s = store.filter_stats()
    assert "stored" in s
    assert "discarded" in s
    assert "compression_ratio" in s
    assert "currently_stored" in s


def test_clear():
    store = OtuxStore(dim=8, mode="full")
    store.write("x", make_vec(8))
    store.clear()
    assert len(store) == 0


# ── Vector validation ──────────────────────────────────────────────────────

def test_wrong_dim_raises():
    store = OtuxStore(dim=8)
    with pytest.raises(ValueError, match="dim mismatch"):
        store.write("bad", np.ones(16, dtype=np.float32))


def test_zero_norm_raises():
    """v1.3: zero-norm vector should raise, not silently normalize."""
    store = OtuxStore(dim=8)
    with pytest.raises(ValueError, match="zero-norm"):
        store.write("zero", np.zeros(8, dtype=np.float32))


def test_repr():
    store = OtuxStore(dim=8, mode="full")
    assert "OtuxStore" in repr(store)


# ── Constructor validation (v1.3) ──────────────────────────────────────────

def test_bad_dim_raises():
    with pytest.raises(ValueError, match="dim must be > 0"):
        OtuxStore(dim=0)


def test_bad_max_entries_raises():
    with pytest.raises(ValueError, match="max_entries must be > 0"):
        OtuxStore(max_entries=0)


def test_bad_thresholds_raises():
    with pytest.raises(ValueError, match="forget_threshold"):
        OtuxStore(importance_threshold=0.1, forget_threshold=0.5)


# ── Smart eviction (v1.3) ─────────────────────────────────────────────────

def test_eviction_triggers_at_max_entries():
    """When max_entries is reached, oldest/least-important entries are evicted."""
    store = OtuxStore(dim=8, mode="full", max_entries=5)
    for i in range(10):
        store.write(f"evict_{i}", make_vec(8, i), reward=0.5)
    assert len(store) <= 5
    stats = store.filter_stats()
    assert stats["evicted"] > 0
