# tests/conftest.py
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_vec(rng):
    v = rng.standard_normal(16).astype("float32")
    return v / (np.linalg.norm(v) + 1e-10)


@pytest.fixture
def full_store():
    from zpower.memory.otux import OtuxStore
    store = OtuxStore(dim=16, mode="full")
    rng_  = np.random.default_rng(0)
    for i in range(10):
        v = rng_.standard_normal(16).astype("float32")
        store.write(f"tok_{i}", v, x=i, y=i % 3, z=i % 4)
    return store
