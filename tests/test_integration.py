# tests/test_integration.py  —  Full pipeline, no torch required
import numpy as np
import pytest
import zpower as zp
from zpower.memory.otux             import OtuxStore
from zpower.stabilize.model_stabilizer import ModelStabilizer
from zpower.monitor.nipgraph        import NipGraph
from zpower.weights.vault           import WeightVault
from zpower.weights.surgeon         import WeightSurgeon
from zpower.compute.safe_math       import SafeMath, safe_loss, safe_divide
from zpower.heal                    import AutoHeal


def test_import_and_version():
    assert zp.__version__ == "1.3.0"
    assert zp.__author__  == "NNN Bhoi"


def test_info_no_crash(capsys):
    zp.info()
    captured = capsys.readouterr()
    assert "ZPower" in captured.out
    assert "1.3.0" in captured.out


# ── Full numpy pipeline ────────────────────────────────────────────────────

def test_full_otux_pipeline():
    store = OtuxStore(dim=16, mode="selective")
    rng   = np.random.default_rng(0)

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

    vault = WeightVault(vault_threshold=0.0)
    vault.record(FakeModel(0), {
        "loss": 0.1, "loss_reference": 2.0,
        "val_accuracy": 0.95, "confidence": 0.90,
        "grad_health": "healthy", "curvature": "flat",
    })
    assert vault.summary()["total_snapshots"] > 0

    surgeon = WeightSurgeon(conflict_resolution="highest_performer")
    sd1 = {"fc1": FakeModel(0).fc1, "fc2": FakeModel(0).fc2}
    sd2 = {"fc1": FakeModel(1).fc1, "fc2": FakeModel(1).fc2}
    surgeon.add_source(sd1, label="model_a", perf_score=0.6)
    surgeon.add_source(sd2, label="model_b", perf_score=0.9)
    best = surgeon.select_best()
    assert "fc1" in best
    assert "fc2" in best


def test_safe_math_pipeline():
    p = np.array([0.3, 0.3, 0.4])
    t = np.array([0.0, 0.0, 1.0])
    loss = safe_loss(p, t, "mse")
    assert loss > 0

    token = safe_divide(1, 3)
    assert abs(float(token) - 1/3) < 1e-9

    token2 = safe_divide(1, 3)
    assert token.name == token2.name


def test_nipgraph_training_simulation():
    ng = NipGraph(variables=["loss", "accuracy"])

    for i in range(30):
        loss = 2.0 * np.exp(-0.1 * i) + 0.01 * np.random.default_rng(i).standard_normal()
        acc  = 1.0 - loss / 2.0
        ng.update("loss",     max(loss, 0.01))
        ng.update("accuracy", min(max(acc, 0.0), 1.0))

    state = ng.check()
    assert "loss" in state
    assert "accuracy" in state
    assert isinstance(state["loss"]["track"], str)


# ── AutoHeal tests (v1.3) ──────────────────────────────────────────────────

def test_autoheal_continue_on_healthy():
    """AutoHeal returns CONTINUE when everything is healthy."""
    healer = AutoHeal(model=None, vault=None, optimizer=None)
    result = healer.on_step(loss_value=0.5, grad_state="healthy",
                            stability_info={"state": "healthy"})
    assert result == "continue"


def test_autoheal_skip_on_nan():
    """AutoHeal returns SKIP when loss is NaN."""
    healer = AutoHeal(model=None, vault=None, optimizer=None)
    result = healer.on_step(loss_value=float("nan"), grad_state="healthy")
    assert result == "skip"


def test_autoheal_strategy_validation():
    """Invalid strategy raises ValueError."""
    with pytest.raises(ValueError, match="invalid strategy"):
        AutoHeal(model=None, vault=None, optimizer=None, strategy="bad_strategy")


def test_autoheal_valid_strategies():
    """All valid strategies can be created."""
    for strategy in ("both", "rollback_only", "lr_only", "restart"):
        healer = AutoHeal(model=None, vault=None, optimizer=None, strategy=strategy)
        assert healer.strategy == strategy


def test_autoheal_reset():
    """reset() clears heal state."""
    healer = AutoHeal(model=None, vault=None, optimizer=None, max_heals=10)
    healer._heal_count = 3
    healer._explode_streak = 2
    healer.reset()
    assert healer._heal_count == 0
    assert healer._explode_streak == 0
    assert healer._heal_log == []


def test_autoheal_max_heals_raises():
    """Exceeding max_heals raises RuntimeError."""
    healer = AutoHeal(model=None, vault=None, optimizer=None, max_heals=1)
    healer._heal_count = 1
    with pytest.raises(RuntimeError, match="Max heal events"):
        healer.on_step(loss_value=0.5, grad_state="healthy",
                       stability_info={"state": "diverging"})


# ── Repr tests (v1.3) ─────────────────────────────────────────────────────

def test_repr_methods():
    """All major classes have useful __repr__."""
    store = OtuxStore(dim=8, mode="full")
    assert "OtuxStore" in repr(store)

    from zpower.stabilize.grad_shield import GradShield
    gs = GradShield()
    assert "GradShield" in repr(gs)

    from zpower.stabilize.stability_core import StabilityCore
    sc = StabilityCore()
    assert "StabilityCore" in repr(sc)

    from zpower.compute.safe_math import SafeMath
    sm = SafeMath()
    assert "SafeMath" in repr(sm)

    from zpower.utils.config import ZPConfig
    cfg = ZPConfig()
    assert "ZPConfig" in repr(cfg)


# ── Config validation tests (v1.3) ─────────────────────────────────────────

def test_config_validate_raises_on_bad_values():
    """ZPConfig.validate() raises ValueError, not assert, on bad values."""
    from zpower.utils.config import ZPConfig
    with pytest.raises(ValueError):
        ZPConfig(otux_dim=-1).validate()
    with pytest.raises(ValueError):
        ZPConfig(vault_threshold=0).validate()
    with pytest.raises(ValueError):
        ZPConfig(heal_strategy="invalid").validate()
    with pytest.raises(ValueError):
        ZPConfig(stability_ema_beta=0).validate()


# ── Health API consistency ─────────────────────────────────────────────────

def test_health_api_consistency():
    """All core classes have both health() and status() methods returning dicts."""
    classes_to_test = [
        OtuxStore(dim=8, mode="full"),
        ModelStabilizer(),
        NipGraph(variables=["loss"]),
        WeightVault(),
        SafeMath(),
    ]
    for obj in classes_to_test:
        h = obj.health()
        s = obj.status()
        assert isinstance(h, dict), f"{type(obj).__name__}.health() should return dict"
        assert isinstance(s, dict), f"{type(obj).__name__}.status() should return dict"
        assert "status" in h, f"{type(obj).__name__}.health() should have 'status' key"
