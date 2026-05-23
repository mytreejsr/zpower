# tests/test_weights.py
import numpy as np
import pytest
from zpower.weights.vault   import WeightVault, VaultSnapshot
from zpower.weights.surgeon import WeightSurgeon
from zpower.weights.guard   import WeightGuard


# ── WeightVault ────────────────────────────────────────────────────────────

def _make_numpy_model():
    class FakeModel:
        def __init__(self):
            self.layer1 = np.random.randn(4, 4).astype(np.float32)
            self.layer2 = np.random.randn(4, 2).astype(np.float32)
    return FakeModel()


def _good_metrics():
    return {
        "loss":          0.2,
        "loss_reference": 2.0,
        "val_accuracy":  0.92,
        "confidence":    0.88,
        "grad_health":   "healthy",
        "curvature":     "flat",
    }


def _bad_metrics():
    return {
        "loss":          9.5,
        "loss_reference": 10.0,
        "val_accuracy":  0.2,
        "confidence":    0.1,
        "grad_health":   "exploding",
        "curvature":     "sharp",
    }


def test_vault_stores_good_model():
    vault = WeightVault(vault_threshold=0.75)
    model = _make_numpy_model()
    stored = vault.record(model, _good_metrics())
    assert stored is True
    assert vault.summary()["total_snapshots"] > 0


def test_vault_rejects_bad_model():
    vault = WeightVault(vault_threshold=0.75)
    model = _make_numpy_model()
    stored = vault.record(model, _bad_metrics())
    assert stored is False
    assert vault.summary()["total_snapshots"] == 0


def test_vault_max_per_layer():
    vault = WeightVault(vault_threshold=0.0, max_per_layer=3)
    model = _make_numpy_model()
    metrics = _good_metrics()
    for _ in range(6):
        vault.record(model, metrics)
    for snaps in vault._snapshots.values():
        assert len(snaps) <= 3


def test_get_best_none_when_empty():
    vault = WeightVault()
    assert vault.get_best("layer1") is None


def test_get_best_returns_snapshot():
    vault = WeightVault(vault_threshold=0.0)
    model = _make_numpy_model()
    vault.record(model, _good_metrics())
    snap = vault.get_best("layer1")
    assert snap is not None
    assert isinstance(snap, VaultSnapshot)
    assert snap.perf_score > 0


def test_summary_keys():
    vault = WeightVault()
    s = vault.summary()
    for key in ("layers_vaulted", "total_snapshots", "epochs_evaluated"):
        assert key in s


def test_vault_save_load(tmp_path):
    vault = WeightVault(vault_threshold=0.0)
    model = _make_numpy_model()
    vault.record(model, _good_metrics())

    save_path = str(tmp_path / "vault")
    vault.save(save_path)

    vault2 = WeightVault()
    vault2.load(save_path)
    assert vault2.summary()["total_snapshots"] == vault.summary()["total_snapshots"]


def test_vault_repr():
    vault = WeightVault()
    assert "WeightVault" in repr(vault)


def test_vault_snapshot_repr():
    snap = VaultSnapshot(layer_name="test", epoch=1, weights=np.zeros(4), metrics={}, perf_score=0.9)
    assert "VaultSnapshot" in repr(snap)
    assert "0.9000" in repr(snap)


def test_get_snapshot_metrics():
    """v1.3: get_snapshot_metrics() returns metrics for all snapshots of a layer."""
    vault = WeightVault(vault_threshold=0.0, max_per_layer=3)
    model = _make_numpy_model()
    vault.record(model, _good_metrics())
    vault.record(model, _good_metrics())
    metrics_list = vault.get_snapshot_metrics("layer1")
    assert len(metrics_list) >= 1
    assert "perf_score" in metrics_list[0]
    assert "loss" in metrics_list[0]


def test_vault_bad_threshold_raises():
    with pytest.raises(ValueError, match="vault_threshold"):
        WeightVault(vault_threshold=-0.1)


def test_vault_file_not_found():
    vault = WeightVault()
    with pytest.raises(FileNotFoundError):
        vault.load("/nonexistent/path/vault.npz")


# ── WeightSurgeon ──────────────────────────────────────────────────────────

def _make_state_dict(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "layer1": rng.standard_normal((4, 4)).astype(np.float32),
        "layer2": rng.standard_normal((4, 2)).astype(np.float32),
    }


def test_surgeon_single_source():
    surgeon = WeightSurgeon()
    surgeon.add_source(_make_state_dict(0), label="m1", perf_score=0.8)
    result = surgeon.select_best()
    assert "layer1" in result
    assert "layer2" in result


def test_surgeon_two_sources_selects_better():
    surgeon = WeightSurgeon(conflict_resolution="highest_performer")
    surgeon.add_source(_make_state_dict(0), label="weak",  perf_score=0.4)
    surgeon.add_source(_make_state_dict(1), label="strong", perf_score=0.9)
    result  = surgeon.select_best()
    report  = surgeon.selection_report()
    assert all(v in ("strong", "blended", "sign_vote(strong)") or "strong" in v
               for v in report.values())


def test_surgeon_weighted_average():
    surgeon = WeightSurgeon(conflict_resolution="weighted_average")
    surgeon.add_source(_make_state_dict(0), label="m1", perf_score=0.5)
    surgeon.add_source(_make_state_dict(1), label="m2", perf_score=0.5)
    result = surgeon.select_best()
    assert result["layer1"].shape == (4, 4)


def test_surgeon_no_sources_raises():
    surgeon = WeightSurgeon()
    with pytest.raises(RuntimeError):
        surgeon.select_best()


def test_surgeon_selection_report_populated():
    surgeon = WeightSurgeon()
    surgeon.add_source(_make_state_dict(0), label="m1", perf_score=0.7)
    surgeon.add_source(_make_state_dict(1), label="m2", perf_score=0.8)
    surgeon.select_best()
    report = surgeon.selection_report()
    assert len(report) == 2


def test_surgeon_repr():
    surgeon = WeightSurgeon()
    assert "WeightSurgeon" in repr(surgeon)
