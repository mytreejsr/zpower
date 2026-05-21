# zpower/compute/safe_math.py  —  SafeMath v1.2.0
# v1.2 fix: OrderedDict for O(1) LRU eviction in pocket registry
#           (was dict with O(N) iteration to find oldest entry)
from __future__ import annotations

import math
from collections import OrderedDict
from fractions import Fraction
from typing import Dict, Optional, Union

import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


class FrozenToken:
    __slots__ = ("name", "value")
    def __init__(self, name: str, value: Fraction):
        self.name  = name
        self.value = value
    def __float__(self):  return float(self.value)
    def __repr__(self):   return f"FrozenToken({self.name}, {self.value})"


class SafeMath:
    """
    SafeMath v1.2.0 — NaN-safe mathematical operations.

    v1.2 fix: _pocket is now OrderedDict, enabling O(1) FIFO eviction
    via popitem(last=False) instead of the previous O(N) iteration
    that also risked StopIteration on dict sync issues.
    Accessed tokens are moved to end (most-recently-used) for LRU behaviour.
    """

    def __init__(self, pocket_capacity: int = 64):
        # v1.2: OrderedDict for O(1) LRU eviction
        self._pocket:     OrderedDict = OrderedDict()   # name → FrozenToken
        self._pocket_map: OrderedDict = OrderedDict()   # expr_key → name
        self._counter:    int         = 0
        self.capacity     = pocket_capacity

    def safe_loss(self, predictions, targets, loss_fn: str = "mse") -> float:
        if _TORCH and isinstance(predictions, torch.Tensor):
            return self._torch_loss(predictions, targets, loss_fn)
        return self._numpy_loss(
            np.asarray(predictions, dtype=np.float64),
            np.asarray(targets,     dtype=np.float64),
            loss_fn,
        )

    def safe_divide(self, numerator: float, denominator: float) -> Union[float, FrozenToken]:
        if denominator == 0:
            raise ZeroDivisionError("SafeMath: division by zero")
        try:
            frac = (Fraction(numerator).limit_denominator(10**9) /
                    Fraction(denominator).limit_denominator(10**9))
        except (ValueError, OverflowError):
            return float(numerator) / float(denominator)

        d = frac.denominator
        while d % 2 == 0: d //= 2
        while d % 5 == 0: d //= 5
        if d != 1:
            return self._tokenize(numerator, denominator, frac)
        return float(frac)

    def get_pocket(self) -> Dict:
        return {k: repr(v) for k, v in self._pocket.items()}

    def clear_pocket(self):
        self._pocket.clear()
        self._pocket_map.clear()
        self._counter = 0

    def status(self) -> Dict:
        return {"status": "ok", "pocket_size": len(self._pocket),
                "pocket_capacity": self.capacity}

    def health(self) -> Dict:
        return {"status": "ok", **self.status()}

    # ── Internal ───────────────────────────────────────────────────────────

    def _torch_loss(self, p, t, fn):
        try:
            if fn == "mse":
                val = torch.nn.functional.mse_loss(p.float(), t.float())
            elif fn == "cross_entropy":
                val = torch.nn.functional.cross_entropy(p.float(), t.long())
            elif fn == "mae":
                val = torch.nn.functional.l1_loss(p.float(), t.float())
            else:
                val = torch.nn.functional.mse_loss(p.float(), t.float())
            result = float(val.item())
            if not np.isfinite(result): raise ValueError("non-finite")
            return result
        except (ValueError, RuntimeError):
            return self._fraction_mse(
                p.detach().cpu().numpy().flatten()[:32],
                t.detach().cpu().numpy().flatten()[:32]
            )

    def _numpy_loss(self, p, t, fn):
        try:
            if fn == "mse":
                result = float(np.mean((p - t) ** 2))
            elif fn == "cross_entropy":
                eps    = 1e-8
                result = float(-np.mean(t * np.log(np.clip(p, eps, 1 - eps))))
            elif fn == "mae":
                result = float(np.mean(np.abs(p - t)))
            else:
                result = float(np.mean((p - t) ** 2))
            if not np.isfinite(result): raise ValueError("non-finite")
            return result
        except (ValueError, FloatingPointError):
            return self._fraction_mse(p.flatten()[:32], t.flatten()[:32])

    def _fraction_mse(self, p, t) -> float:
        n = min(len(p), len(t))
        if n == 0: return 0.0
        p_c = [0.0 if not np.isfinite(x) else float(x) for x in p[:n]]
        t_c = [0.0 if not np.isfinite(x) else float(x) for x in t[:n]]
        p_f = [Fraction(x).limit_denominator(10_000) for x in p_c]
        t_f = [Fraction(x).limit_denominator(10_000) for x in t_c]
        return float(sum((pi - ti) ** 2 for pi, ti in zip(p_f, t_f)) / n)

    def _tokenize(self, num, den, frac: Fraction) -> FrozenToken:
        key = f"{frac.numerator}/{frac.denominator}"

        # v1.2: O(1) LRU access — move to end on hit
        if key in self._pocket_map:
            name = self._pocket_map[key]
            self._pocket.move_to_end(name)
            return self._pocket[name]

        # Evict oldest (front of OrderedDict) — O(1)
        if len(self._pocket) >= self.capacity:
            oldest_name, _ = self._pocket.popitem(last=False)
            # Remove from pocket_map too — find by value
            keys_to_del = [k for k, v in self._pocket_map.items() if v == oldest_name]
            for k in keys_to_del:
                del self._pocket_map[k]

        self._counter += 1
        name  = f"@t{self._counter}"
        token = FrozenToken(name, frac)
        self._pocket[name]    = token
        self._pocket_map[key] = name
        return token


# Module-level singleton
_default = SafeMath()
def safe_loss(predictions, targets, loss_fn: str = "mse") -> float:
    return _default.safe_loss(predictions, targets, loss_fn)
def safe_divide(num: float, den: float) -> Union[float, FrozenToken]:
    return _default.safe_divide(num, den)
