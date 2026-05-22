# tests/test_stabilize.py
import numpy as np
import pytest
from zpower.stabilize.grad_shield    import GradShield
from zpower.stabilize.stability_core import StabilityCore
from zpower.stabilize.model_stabilizer import ModelStabilizer


# ── GradShield ─────────────────────────────────────────────────────────────

def test_healthy_gradient():
    gs = GradShield(clip_norm=5.0, vanish_thresh=1e-7, explode_thresh=50.0)
    g  = np.ones(4, dtype=np.float32) * 0.5   # norm ≈ 1.0
    assert gs.check(g) == GradShield.HEALTHY


def test_vanishing_gradient():
    gs = GradShield(vanish_thresh=1e-7)
    g  = np.ones(4, dtype=np.float32) * 1e-9
    assert gs.check(g) == GradShield.VANISHING


def test_warning_gradient():
    gs = GradShield(clip_norm=5.0, explode_thresh=50.0)
    # norm = 10.0 → above clip_norm, below explode → WARNING
    g  = np.ones(4, dtype=np.float32) * 5.1
    assert gs.check(g) == GradShield.WARNING


def test_exploding_gradient():
    gs = GradShield(explode_thresh=50.0)
    g  = np.ones(4, dtype=np.float32) * 100.0
    assert gs.check(g) == GradShield.EXPLODING


def test_shield_clips_warning():
    gs = GradShield(clip_norm=5.0)
    g  = np.ones(4, dtype=np.float32) * 10.0   # norm = 20
    shielded = gs.shield(g)
    clipped_norm = float(np.linalg.norm(shielded))
    assert clipped_norm <= 5.0 + 1e-5


def test_shield_leaves_healthy_unchanged():
    gs = GradShield(clip_norm=5.0)
    g  = np.ones(4, dtype=np.float32) * 0.5    # norm = 1.0
    shielded = gs.shield(g)
    assert np.allclose(g, shielded)


def test_history_recorded():
    gs = GradShield()
    gs.shield(np.ones(4, dtype=np.float32))
    gs.shield(np.ones(4, dtype=np.float32))
    assert len(gs.get_history()) == 2


def test_status_keys():
    gs = GradShield()
    gs.shield(np.ones(4))
    s = gs.status()
    assert "health_rate" in s
    assert "last_state" in s


# ── StabilityCore ──────────────────────────────────────────────────────────

def test_ema_decreasing():
    sc = StabilityCore(ema_beta=0.9)
    losses = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
    last_ema = 2.0
    for L in losses:
        info = sc.update(L)
    assert info["ema"] < 2.0


def test_plateau_detection():
    sc = StabilityCore(
        ema_beta=0.95,
        plateau_steps=5,
        plateau_threshold=0.001,
    )
    # Feed constant loss → triggers plateau
    for _ in range(15):
        info = sc.update(1.0000)
    assert info["plateau_detected"] is True
    assert info["lr_signal"] < 1.0


def test_healthy_state_no_plateau():
    sc = StabilityCore(plateau_steps=10)
    info = sc.update(5.0)
    assert info["state"] in ("healthy", "plateau")


def test_flat_minima():
    sc = StabilityCore(curvature_window=20)
    for _ in range(25):
        sc.update(0.0001)
    assert sc.is_flat_minima() is True


def test_sharp_minima():
    sc = StabilityCore(curvature_window=20)
    import random
    rng = random.Random(0)
    for _ in range(25):
        sc.update(rng.uniform(0, 10))
    assert sc._curvature() == "sharp"


def test_reset():
    sc = StabilityCore()
    for _ in range(5): sc.update(1.0)
    sc.reset()
    assert sc._step == 0
    assert sc._ema  == 0.0


def test_update_returns_required_keys():
    sc   = StabilityCore()
    info = sc.update(1.5)
    for key in ("ema", "raw", "state", "curvature", "plateau_detected", "lr_signal"):
        assert key in info


# ── ModelStabilizer ────────────────────────────────────────────────────────

def test_model_stabilizer_on_loss():
    ms   = ModelStabilizer()
    info = ms.on_loss(1.0)
    assert "ema" in info
    assert "state" in info


def test_model_stabilizer_status():
    ms = ModelStabilizer()
    s  = ms.status()
    assert "grad_shield" in s
    assert "stability_core" in s
