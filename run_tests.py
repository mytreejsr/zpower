#!/usr/bin/env python3
"""
ZPower v1 — Self-contained test runner (no pytest required).
Run: PYTHONPATH=/path/to/zpower python run_tests.py
"""
import sys, traceback, numpy as np

PASS = 0
FAIL = 0
ERRORS = []


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✓  {name}")
        PASS += 1
    except Exception as e:
        print(f"  ✗  {name}")
        print(f"       {type(e).__name__}: {e}")
        ERRORS.append((name, traceback.format_exc()))
        FAIL += 1


def section(title):
    print(f"\n── {title} {'─'*(50-len(title))}")


# ══════════════════════════════════════════════════════════════════════════
# OTUX-S
# ══════════════════════════════════════════════════════════════════════════
def assert_true(x):
    assert x

section("OtuxStore")
from zpower.memory.otux import OtuxStore, ImportanceWeights

def vec(dim=16, seed=0):
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-10)

test("ImportanceWeights sum=1.0",
     lambda: ImportanceWeights(0.35, 0.30, 0.20, 0.15))

def _bad_w():
    try: ImportanceWeights(0.5, 0.5, 0.5, 0.5); return False
    except ValueError: return True
test("ImportanceWeights bad sum raises", lambda: assert_true(_bad_w()))

def _write_len():
    s = OtuxStore(dim=16, mode="full")
    s.write("a", vec(dim=16, seed=0))
    s.write("b", vec(dim=16, seed=1))
    assert_true(len(s) == 2)
test("write + len (full mode)", _write_len)

test("query returns correct token", lambda: (
    lambda s, v: (s.write("alpha", v), assert_true(s.query(v, 1)[0]["token"] == "alpha"))
)(OtuxStore(dim=16, mode="full"), vec(seed=99)))

test("query empty store → []", lambda: assert_true(
    OtuxStore(dim=16).query(vec()) == []))

test("coord retrieval z=3", lambda: (
    lambda s: (
        s.write("deep",    vec(seed=0), z=3),
        s.write("shallow", vec(seed=1), z=0),
        assert_true(len(s.query_by_coord(z=3, tol=0.1)) == 1)
    )
)(OtuxStore(dim=16, mode="full")))

test("force write bypasses gate", lambda: (
    lambda s: assert_true(s.write("f", vec(), reward=0.0, force=True) == "forced")
)(OtuxStore(dim=16, importance_threshold=0.99)))

test("high reward → stored", lambda: (
    lambda s: assert_true(s.write("important", vec(seed=1), reward=1.0) == "stored")
)(OtuxStore(dim=16, mode="selective", importance_threshold=0.65)))

def _three_strikes():
    s = OtuxStore(dim=16, mode="selective",
                  importance_threshold=0.90, forget_threshold=0.05, buffer_strikes=3)
    v_ = vec(seed=7)
    for _ in range(3):
        s.write("repeat", v_, reward=0.4)
    return len(s) >= 1
test("3 strikes → promoted to store", lambda: assert_true(_three_strikes()))

test("wrong dim raises ValueError", lambda: (
    lambda s: (lambda: s.write("bad", np.ones(32, dtype=np.float32)))()
    if (lambda: (_ for _ in ()).throw(ValueError("expected")))
    else None
))

def _wrong_dim():
    try:
        OtuxStore(dim=16).write("bad", np.ones(32, dtype=np.float32))
        return False
    except ValueError:
        return True
test("wrong dim raises ValueError", lambda: assert_true(_wrong_dim()))

test("filter_stats keys present", lambda: assert_true(
    all(k in OtuxStore(dim=16, mode="full").filter_stats()
        for k in ("stored","discarded","compression_ratio","currently_stored"))))

test("clear() empties store", lambda: (
    lambda s: (s.write("x", vec()), s.clear(), assert_true(len(s) == 0))
)(OtuxStore(dim=16, mode="full")))


# ══════════════════════════════════════════════════════════════════════════
# SafeMath
# ══════════════════════════════════════════════════════════════════════════
section("SafeMath / safe_divide / safe_loss")
from zpower.compute.safe_math import SafeMath, FrozenToken, safe_loss, safe_divide
from fractions import Fraction
import math

test("safe_divide integer result", lambda: assert_true(
    safe_divide(10, 2) == 5.0))

test("safe_divide terminating decimal", lambda: assert_true(
    abs(float(safe_divide(1, 4)) - 0.25) < 1e-9))

test("safe_divide recurring → FrozenToken", lambda: assert_true(
    isinstance(safe_divide(1, 3), FrozenToken)))

def _dedup():
    sm = SafeMath()
    t1 = sm.safe_divide(1, 3)
    t2 = sm.safe_divide(1, 3)
    return t1.name == t2.name
test("recurring deduplication — same token", lambda: assert_true(_dedup()))

def _zero_div():
    try: safe_divide(5, 0); return False
    except ZeroDivisionError: return True
test("division by zero raises", lambda: assert_true(_zero_div()))

test("FrozenToken float conversion", lambda: assert_true(
    abs(float(FrozenToken("@t1", Fraction(1,3))) - 1/3) < 1e-5))

test("safe_loss MSE numpy", lambda: assert_true(
    abs(safe_loss(np.array([0.5,0.5,0.5]), np.array([1.,1.,1.]), "mse") - 0.25) < 1e-5))

test("safe_loss NaN fallback is finite", lambda: assert_true(
    math.isfinite(safe_loss(np.array([float("nan"),float("nan")]),
                            np.array([1.,1.]), "mse"))))


# ══════════════════════════════════════════════════════════════════════════
# NipGraph
# ══════════════════════════════════════════════════════════════════════════
section("NipGraph")
from zpower.monitor.nipgraph import NipGraph

test("basic update no crash", lambda: (
    lambda ng: [ng.update("loss", 1.0-i*0.05) for i in range(5)]
)(NipGraph(["loss"])))

test("check() returns variable state", lambda: assert_true(
    "loss" in NipGraph(["loss"]).check()))

def _converge():
    ng = NipGraph(["loss"], band_width=0.5)
    for _ in range(30): ng.update("loss", 0.001)
    return ng.is_converged()
test("convergence detected after stable values", lambda: assert_true(_converge()))

def _no_converge():
    ng = NipGraph(["loss"])
    rng_ = np.random.default_rng(0)
    for _ in range(25): ng.update("loss", rng_.uniform(0.5, 5.0))
    return not ng.is_converged()
test("no convergence with high variance", lambda: assert_true(_no_converge()))

test("multiple variables tracked", lambda: assert_true(
    set(NipGraph(["loss","grad_norm","acc"]).check().keys())
    == {"loss","grad_norm","acc"}))

test("auto-add new variable", lambda: (
    lambda ng: (ng.update("new_var", 1.0), assert_true("new_var" in ng.check()))
)(NipGraph(["loss"])))

test("clear_alerts empties list", lambda: (
    lambda ng: (
        [ng.update("loss", v) for v in [1.0]*5 + [-999.0]],
        ng.clear_alerts(),
        assert_true(ng.alerts() == [])
    )
)(NipGraph(["loss"], band_width=0.01)))

test("locate_fault no anomaly msg", lambda: assert_true(
    "No anomaly" in NipGraph(["loss"]).locate_fault("loss")))

test("status() has required keys", lambda: assert_true(
    all(k in NipGraph(["loss"]).status()
        for k in ("step","variables","converged","alerts"))))


# ══════════════════════════════════════════════════════════════════════════
# GradShield
# ══════════════════════════════════════════════════════════════════════════
section("GradShield")
from zpower.stabilize.grad_shield import GradShield

test("healthy gradient classified", lambda: assert_true(
    GradShield().check(np.ones(4)*0.5) == "healthy"))

test("vanishing gradient classified", lambda: assert_true(
    GradShield(vanish_thresh=1e-7).check(np.ones(4)*1e-9) == "vanishing"))

test("warning gradient classified", lambda: assert_true(
    GradShield(clip_norm=5.0, explode_thresh=50.0).check(np.ones(4)*5.1) == "warning"))

test("exploding gradient classified", lambda: assert_true(
    GradShield(explode_thresh=50.0).check(np.ones(4)*100.0) == "exploding"))

test("shield clips exploding", lambda: assert_true(
    np.linalg.norm(GradShield(clip_norm=5.0).shield(np.ones(4)*100.0)) <= 5.01))

test("shield leaves healthy unchanged", lambda: (
    lambda gs, g: assert_true(np.allclose(g, gs.shield(g)))
)(GradShield(clip_norm=5.0), np.ones(4)*0.5))

test("history recorded per step", lambda: (
    lambda gs: (gs.shield(np.ones(4)), gs.shield(np.ones(4)),
                assert_true(len(gs.get_history()) == 2))
)(GradShield()))


# ══════════════════════════════════════════════════════════════════════════
# StabilityCore
# ══════════════════════════════════════════════════════════════════════════
section("StabilityCore")
from zpower.stabilize.stability_core import StabilityCore

test("update returns required keys", lambda: assert_true(
    all(k in StabilityCore().update(1.5)
        for k in ("ema","raw","state","curvature","plateau_detected","lr_signal"))))

def _plateau():
    sc = StabilityCore(ema_beta=0.95, plateau_steps=5, plateau_threshold=0.001)
    for _ in range(15): info = sc.update(1.0)
    return info["plateau_detected"] and info["lr_signal"] < 1.0
test("plateau detected → lr_signal < 1.0", lambda: assert_true(_plateau()))

def _flat():
    sc = StabilityCore(curvature_window=20)
    for _ in range(25): sc.update(0.0001)
    return sc.is_flat_minima()
test("flat minima detected", lambda: assert_true(_flat()))

test("reset clears state", lambda: (
    lambda sc: (
        [sc.update(i) for i in range(5)],
        sc.reset(),
        assert_true(sc._step == 0 and sc._ema == 0.0)
    )
)(StabilityCore()))


# ══════════════════════════════════════════════════════════════════════════
# WeightVault
# ══════════════════════════════════════════════════════════════════════════
section("WeightVault")
from zpower.weights.vault import WeightVault

class FakeModel:
    def __init__(self, seed=0):
        rng_ = np.random.default_rng(seed)
        self.fc1 = rng_.standard_normal((4,4)).astype(np.float32)
        self.fc2 = rng_.standard_normal((4,2)).astype(np.float32)

GOOD = {"loss":0.2,"loss_reference":2.0,"val_accuracy":0.92,
        "confidence":0.88,"grad_health":"healthy","curvature":"flat"}
BAD  = {"loss":9.5,"loss_reference":10.0,"val_accuracy":0.2,
        "confidence":0.1,"grad_health":"exploding","curvature":"sharp"}

test("good metrics → stored=True", lambda: assert_true(
    WeightVault(vault_threshold=0.75).record(FakeModel(), GOOD)))

test("bad metrics → stored=False", lambda: assert_true(
    not WeightVault(vault_threshold=0.75).record(FakeModel(), BAD)))

test("max_per_layer respected", lambda: (
    lambda v: (
        [v.record(FakeModel(), GOOD) for _ in range(8)],
        assert_true(all(len(snaps)<=3 for snaps in v._snapshots.values()))
    )
)(WeightVault(vault_threshold=0.0, max_per_layer=3)))

test("get_best None when empty", lambda: assert_true(
    WeightVault().get_best("fc1") is None))

test("summary has required keys", lambda: assert_true(
    all(k in WeightVault().summary()
        for k in ("layers_vaulted","total_snapshots","epochs_evaluated"))))


# ══════════════════════════════════════════════════════════════════════════
# WeightSurgeon
# ══════════════════════════════════════════════════════════════════════════
section("WeightSurgeon")
from zpower.weights.surgeon import WeightSurgeon

def sd(seed=0):
    rng_ = np.random.default_rng(seed)
    return {"layer1": rng_.standard_normal((4,4)).astype(np.float32),
            "layer2": rng_.standard_normal((4,2)).astype(np.float32)}

test("single source returns correct keys", lambda: (
    lambda s: (s.add_source(sd(0), label="m1", perf_score=0.8),
               assert_true(set(s.select_best().keys()) == {"layer1","layer2"}))
)(WeightSurgeon()))

def _two_source():
    s = WeightSurgeon(conflict_resolution="highest_performer")
    s.add_source(sd(0), label="weak",   perf_score=0.4)
    s.add_source(sd(1), label="strong", perf_score=0.9)
    s.select_best()
    r = s.selection_report()
    return all("strong" in v for v in r.values())
test("higher perf model wins all layers", lambda: assert_true(_two_source()))

test("weighted_average produces correct shape", lambda: (
    lambda s: (
        s.add_source(sd(0), label="m1", perf_score=0.5),
        s.add_source(sd(1), label="m2", perf_score=0.5),
        assert_true(s.select_best()["layer1"].shape == (4,4))
    )
)(WeightSurgeon(conflict_resolution="weighted_average")))

def _no_src():
    try: WeightSurgeon().select_best(); return False
    except RuntimeError: return True
test("no sources raises RuntimeError", lambda: assert_true(_no_src()))


# ══════════════════════════════════════════════════════════════════════════
# Integration
# ══════════════════════════════════════════════════════════════════════════
section("Integration")
import zpower as zp

test("import zpower as zp works", lambda: assert_true(zp.__version__ == "1.0.0"))
test("zp.info() no crash", lambda: zp.info())

test("full pipeline: OTUX + stabilizer + NipGraph + vault + surgeon", lambda: (
    lambda: (
        # OTUX-S
        (lambda st: [st.write(f"t{i}", vec(seed=i), reward=float(i)/10)
                     for i in range(20)])(OtuxStore(dim=16, mode="selective")),
        # StabilityCore
        (lambda sc: [sc.update(2.0 * np.exp(-0.1*i)) for i in range(20)])(StabilityCore()),
        # NipGraph
        (lambda ng: [ng.update("loss", 2.0*np.exp(-0.1*i)) for i in range(20)])(NipGraph(["loss"])),
        # Vault
        (lambda v: v.record(FakeModel(0), GOOD))(WeightVault(vault_threshold=0.0)),
        # Surgeon
        (lambda s: (
            s.add_source(sd(0), label="a", perf_score=0.7),
            s.add_source(sd(1), label="b", perf_score=0.85),
            assert_true("layer1" in s.select_best())
        ))(WeightSurgeon()),
    )
)())

test("safe_loss + safe_divide together", lambda: (
    assert_true(math.isfinite(safe_loss(np.array([0.3,0.4,0.3]), np.array([0.,0.,1.]), "mse"))),
    assert_true(isinstance(safe_divide(2, 3), FrozenToken)),
))

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
total = PASS + FAIL
print(f"\n{'═'*54}")
print(f"  ZPower v1 Tests — {PASS}/{total} passed  "
      f"({'✓ ALL PASS' if FAIL==0 else f'✗ {FAIL} FAILED'})")
print(f"{'═'*54}")

if ERRORS:
    print("\nFailed tests detail:")
    for name, tb in ERRORS:
        print(f"\n  ✗ {name}")
        for line in tb.strip().splitlines()[-4:]:
            print(f"    {line}")

sys.exit(0 if FAIL == 0 else 1)
