# tests/test_nipgraph.py
import pytest
from zpower.monitor.nipgraph import NipGraph


def test_basic_update_no_alert():
    ng = NipGraph(variables=["loss"], band_width=0.5)
    for i in range(5):
        ng.update("loss", 1.0 - i * 0.05)
    state = ng.check()
    assert "loss" in state
    assert state["loss"]["alert"] is False


def test_track_classification_positive_stable():
    ng = NipGraph(variables=["loss"], band_width=0.5)
    for _ in range(10):
        ng.update("loss", 1.0)
    state = ng.check()
    assert state["loss"]["track"] in ("x_M", "x_W")


def test_alert_fires_on_bad_jump():
    ng = NipGraph(variables=["loss"], band_width=0.05)
    # Establish stable positive trend
    for _ in range(10):
        ng.update("loss", 1.0)
    # Sudden negative spike (EMA positive, value hugely negative)
    ng.update("loss", -100.0)
    state = ng.check()
    # Track should have jumped to Y_W territory
    alerts = ng.alerts()
    # Either direct alert or track is Y_W
    assert state["loss"]["track"] in ("Y_M", "Y_W") or len(alerts) > 0


def test_convergence_detection():
    ng = NipGraph(variables=["loss"], band_width=0.5)
    # Feed very stable values — should converge
    for _ in range(25):
        ng.update("loss", 0.001)
    assert ng.is_converged() is True


def test_not_converged_with_variance():
    ng = NipGraph(variables=["loss"])
    import random
    rng = random.Random(42)
    for _ in range(25):
        ng.update("loss", rng.uniform(0.5, 5.0))
    assert ng.is_converged() is False


def test_multiple_variables():
    ng = NipGraph(variables=["loss", "grad_norm", "accuracy"])
    ng.update("loss",      2.0)
    ng.update("grad_norm", 0.5)
    ng.update("accuracy",  0.8)
    state = ng.check()
    assert set(state.keys()) == {"loss", "grad_norm", "accuracy"}


def test_auto_add_new_variable():
    ng = NipGraph(variables=["loss"])
    ng.update("new_metric", 1.0)   # not in initial list
    state = ng.check()
    assert "new_metric" in state


def test_clear_alerts():
    ng = NipGraph(variables=["loss"], band_width=0.01)
    for _ in range(5):
        ng.update("loss", 1.0)
    ng.update("loss", -999.0)
    ng.clear_alerts()
    assert ng.alerts() == []


def test_locate_fault_no_anomaly():
    ng = NipGraph(variables=["loss"])
    ng.update("loss", 1.0)
    msg = ng.locate_fault("loss")
    assert "No anomaly" in msg


def test_render_panels_no_crash(capsys):
    ng = NipGraph(variables=["loss", "accuracy"])
    for _ in range(3):
        ng.update("loss",     1.0)
        ng.update("accuracy", 0.9)
    ng.render_panels()
    captured = capsys.readouterr()
    assert "NipGraph" in captured.out


def test_status_keys():
    ng = NipGraph(variables=["loss"])
    s = ng.status()
    assert "step" in s
    assert "variables" in s
    assert "converged" in s
    assert "alerts" in s
