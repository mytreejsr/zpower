# tests/test_integration.py  —  Full pipeline, no torch required
import numpy as np
import pytest
import zpower as zp
from zpower.memory.otux             import OtuxStore
from zpower.stabilize.model_stabilizer import ModelStabilizer
from zpower.monitor.nipgraph        import NipGraph
from zpower.weights.vault           import WeightVault
from zpower.weights.surgeon         import WeightSurgeon
from zpower.compute.safe_math       import safe_loss, safe_divide


def test_import_and_version():
    assert zp.__version__ == "1.0.0"
    assert zp.__author__  == "NNN Bhoi"


def test_info_no_crash(capsys):
    zp.info()
    captured = capsys.readouterr()
    assert "ZPower" in captured.out


# ── Full numpy pipeline ────────────────────────────────────────────────────

def test_full_otux_pipeline():
    store = OtuxStore(dim=16, mode="selective")
    rng   = np.random.default_rng(0)

    # Write several entries
    for i in range(20):
        v = rng.standard_normal(16).astype(np.float32)
        store.write(f"entry_{i}", v, x=i, y=i % 3, z=i % 5, reward=rng.uniform(0, 1))

    stats = store.filter_stats()
    assert stats["total_seen"] == 20
    assert stats["currently_stored"] <= 20


def test_full_stabilizer_pipeline():
    ms = ModelStabilizer()
    ng = NipGraph(variables=["loss", "grad_norm"])
    ms._nipgraph = ng
    ms.stability_core._nipgraph = ng

    losses = [3.0, 2.5, 2.0, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02]
    for L in losses:
        info = ms.on_loss(L)
        assert "state" in info

    # Gradient shield
    g_healthy   = np.ones(4, dtype=np.float32) * 0.5
    g_exploding = np.ones(4, dtype=np.float32) * 100.0
    assert ms.grad_shield.check(g_healthy)   == "healthy"
    assert ms.grad_shield.check(g_exploding) == "exploding"

    shielded = ms.grad_shield.shield(g_exploding)
    assert float(np.linalg.norm(shielded)) <= 5.0 + 1e-4


def test_vault_and_surgeon_pipeline():
    rng = np.random.default_rng(42)

    class FakeModel:
        def __init__(self, seed):
            _rng = np.random.default_rng(seed)
            self.fc1 = _rng.standard_normal((8, 4)).astype(np.float32)
            self.fc2 = _rng.standard_normal((4, 2)).astype(np.float32)

    # Vault: record good model
    vault = WeightVault(vault_threshold=0.0)
    vault.record(FakeModel(0), {
        "loss": 0.1, "loss_reference": 2.0,
        "val_accuracy": 0.95, "confidence": 0.90,
        "grad_health": "healthy", "curvature": "flat",
    })
    assert vault.summary()["total_snapshots"] > 0

    # Surgeon: merge two models
    surgeon = WeightSurgeon(conflict_resolution="highest_performer")
    sd1 = {"fc1": FakeModel(0).fc1, "fc2": FakeModel(0).fc2}
    sd2 = {"fc1": FakeModel(1).fc1, "fc2": FakeModel(1).fc2}
    surgeon.add_source(sd1, label="model_a", perf_score=0.6)
    surgeon.add_source(sd2, label="model_b", perf_score=0.9)
    best = surgeon.select_best()
    assert "fc1" in best
    assert "fc2" in best


def test_safe_math_pipeline():
    # Normal loss
    p = np.array([0.3, 0.3, 0.4])
    t = np.array([0.0, 0.0, 1.0])
    loss = safe_loss(p, t, "mse")
    assert loss > 0

    # Recurring division
    token = safe_divide(1, 3)
    assert abs(float(token) - 1/3) < 1e-9

    # Deduplication
    token2 = safe_divide(1, 3)
    assert token.name == token2.name


def test_nipgraph_training_simulation():
    ng = NipGraph(variables=["loss", "accuracy"])

    # Simulate healthy training
    for i in range(30):
        loss = 2.0 * np.exp(-0.1 * i) + 0.01 * np.random.default_rng(i).standard_normal()
        acc  = 1.0 - loss / 2.0
        ng.update("loss",     max(loss, 0.01))
        ng.update("accuracy", min(max(acc, 0.0), 1.0))

    state = ng.check()
    assert "loss" in state
    assert "accuracy" in state
    # No alerts expected in healthy training
    # (may or may not converge depending on values — just check no crash)
    assert isinstance(state["loss"]["track"], str)
