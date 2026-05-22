# tests/test_safe_math.py
import math
from fractions import Fraction

import numpy as np
import pytest
from zpower.compute.safe_math import SafeMath, FrozenToken, safe_loss, safe_divide


def test_safe_divide_integer():
    sm = SafeMath()
    result = sm.safe_divide(10, 2)
    assert result == 5.0


def test_safe_divide_terminating():
    sm = SafeMath()
    result = sm.safe_divide(1, 4)
    assert abs(float(result) - 0.25) < 1e-9


def test_safe_divide_recurring():
    sm = SafeMath()
    result = sm.safe_divide(10, 3)
    assert isinstance(result, FrozenToken)
    assert result.value == Fraction(10, 3)


def test_safe_divide_deduplication():
    sm = SafeMath()
    t1 = sm.safe_divide(10, 3)
    t2 = sm.safe_divide(10, 3)
    assert t1.name == t2.name     # same pocket token


def test_safe_divide_zero():
    sm = SafeMath()
    with pytest.raises(ZeroDivisionError):
        sm.safe_divide(5, 0)


def test_frozen_token_float():
    t = FrozenToken("@t1", Fraction(1, 3))
    assert abs(float(t) - 0.3333333) < 1e-5


def test_safe_loss_mse_numpy():
    p = np.array([0.5, 0.5, 0.5])
    t = np.array([1.0, 1.0, 1.0])
    loss = safe_loss(p, t, "mse")
    assert abs(loss - 0.25) < 1e-6


def test_safe_loss_mae_numpy():
    p = np.array([0.0, 0.0])
    t = np.array([1.0, 1.0])
    loss = safe_loss(p, t, "mae")
    assert abs(loss - 1.0) < 1e-6


def test_safe_loss_nan_fallback():
    """NaN inputs should not crash — fallback to Fraction arithmetic."""
    p = np.array([float("nan"), float("nan")])
    t = np.array([1.0, 1.0])
    loss = safe_loss(p, t, "mse")
    assert math.isfinite(loss)


def test_pocket_capacity():
    sm = SafeMath(pocket_capacity=3)
    # Fill pocket
    sm.safe_divide(1, 3)
    sm.safe_divide(2, 3)
    sm.safe_divide(4, 3)
    # 4th unique recurring — should evict oldest
    sm.safe_divide(5, 3)
    assert len(sm._pocket) <= 3


def test_get_pocket():
    sm = SafeMath()
    sm.safe_divide(1, 3)
    pocket = sm.get_pocket()
    assert len(pocket) == 1
    assert "@t1" in pocket


def test_clear_pocket():
    sm = SafeMath()
    sm.safe_divide(1, 3)
    sm.clear_pocket()
    assert len(sm.get_pocket()) == 0
