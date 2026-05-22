# zpower/stabilize/stability_core.py  —  StabilityCore v1.2.0
# v1.2 fixes:
#   CRITICAL: plateau detection was comparing EMA vs raw loss (wrong types)
#             Now uses two-window average comparison (valid apples-to-apples)
#   FIX: EMA == 0.0 sentinel replaced with _initialized boolean
#   PERF: history is now deque (was list + manual trim)
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np


class StabilityCore:
    """
    StabilityCore v1.2.0 — Loss landscape monitor.

    v1.2:
      CRITICAL FIX — plateau detection now compares two equal-type windows:
        window_recent  = mean(loss[-K:])
        window_before  = mean(loss[-2K:-K])
        plateau if |window_recent - window_before| < plateau_threshold
      Previously compared EMA (smoothed) vs raw loss value (noisy).

      FIX — _initialized flag replaces `if self._ema == 0.0` sentinel.
      PERF — deque(maxlen) replaces list + manual trim.
    """

    def __init__(
        self,
        ema_beta:          float = 0.95,
        plateau_steps:     int   = 10,
        plateau_threshold: float = 0.001,
        curvature_window:  int   = 20,
        nipgraph           = None,
    ):
        self.beta          = ema_beta
        self.plateau_steps = plateau_steps
        self.plateau_delta = plateau_threshold
        self.curv_window   = curvature_window
        self._nipgraph     = nipgraph

        self._ema:          float = 0.0
        self._initialized:  bool  = False   # v1.2: replaces == 0.0 sentinel
        # v1.2: deque replaces list + trim
        max_hist = max(plateau_steps * 4, curvature_window * 3)
        self._history: deque = deque(maxlen=max_hist)
        self._step:    int   = 0
        self._lr_signal: float = 1.0
        self._plateau_count: int = 0

    # ── Core ───────────────────────────────────────────────────────────────

    def update(self, loss: float) -> Dict:
        val = float(loss)
        self._history.append(val)
        self._step += 1

        # v1.2: _initialized flag — safe when first loss == 0.0
        if not self._initialized:
            self._ema         = val
            self._initialized = True
        else:
            self._ema = self.beta * self._ema + (1 - self.beta) * val

        if self._nipgraph is not None:
            self._nipgraph.update("loss",     val,       step=self._step)
            self._nipgraph.update("loss_ema", self._ema, step=self._step)

        plateau = self._detect_plateau()
        if plateau:
            self._plateau_count += 1
            self._lr_signal = max(0.1, self._lr_signal * 0.5)
        else:
            self._plateau_count = 0
            if val < self._ema * 1.1:
                self._lr_signal = min(1.0, self._lr_signal * 1.05)

        if plateau:
            state = "plateau"
        elif self._initialized and val > self._ema * 2.0 and self._step > 10:
            state = "diverging"
        else:
            state = "healthy"

        return {
            "ema":               round(self._ema, 6),
            "raw":               round(val, 6),
            "state":             state,
            "curvature":         self._curvature(),
            "plateau_detected":  plateau,
            "lr_signal":         round(self._lr_signal, 4),
            "step":              self._step,
        }

    def is_flat_minima(self) -> bool:
        return self._curvature() == "flat"

    def get_lr_signal(self) -> float:
        return self._lr_signal

    def reset(self):
        self._ema          = 0.0
        self._initialized  = False
        self._history.clear()
        self._step         = 0
        self._lr_signal    = 1.0
        self._plateau_count = 0

    def status(self) -> Dict:
        return {
            "status":        "ok",
            "step":          self._step,
            "ema":           round(self._ema, 6),
            "curvature":     self._curvature(),
            "lr_signal":     self._lr_signal,
            "plateau_count": self._plateau_count,
        }

    def health(self) -> Dict:
        return {"status": "ok", **self.status()}

    # ── Internal ───────────────────────────────────────────────────────────

    def _detect_plateau(self) -> bool:
        """
        v1.2 fix: two-window comparison (both are raw loss averages).
        Previously compared EMA (smoothed float) vs raw history value (noisy).
        """
        hist = list(self._history)
        needed = self.plateau_steps * 2
        if len(hist) < needed:
            return False

        window_recent = float(np.mean(hist[-self.plateau_steps:]))
        window_before = float(np.mean(hist[-needed:-self.plateau_steps]))
        return abs(window_recent - window_before) < self.plateau_delta

    def _curvature(self) -> str:
        hist = list(self._history)
        if len(hist) < self.curv_window:
            return "unknown"
        var = float(np.var(hist[-self.curv_window:]))
        if var < 0.01:  return "flat"
        if var < 0.05:  return "moderate"
        return "sharp"
