# zpower/monitor/nipgraph.py  —  NipGraph v1.3.0
# v1.3 changes:
#   API:  Added __repr__
#   FIX:  history list replaced with deque for O(1) trim (was list + pop(0))
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class VarState:
    name:       str
    track:      str   = "x_M"
    state:      str   = "P"
    ema:        float = 0.0
    _initialized: bool = False
    alert:      bool  = False
    alert_step: Optional[int] = None
    history:    deque = field(default_factory=lambda: deque(maxlen=200))


class NipGraph:
    """
    NipGraph v1.3.0 — Parity-aware training anomaly detection.

    v1.2 fixes:
      step counter drift, near-zero EMA false alerts, _initialized flag.

    v1.3:
      VarState.history is now deque(maxlen=200) for O(1) trim.
      Added __repr__.
    """

    def __init__(
        self,
        variables:      List[str],
        band_width:     float = 0.10,
        ema_beta:       float = 0.90,
        history_len:    int   = 200,
        absolute_floor: float = 0.10,
    ):
        self.band           = band_width
        self.beta           = ema_beta
        self.absolute_floor = absolute_floor
        self._vars: Dict[str, VarState] = {}
        for v in variables:
            self._vars[v] = VarState(name=v, history=deque(maxlen=history_len))
        self._history_len   = history_len
        self._step          = 0
        self._alerts: List[Dict] = []

    # ── Update ─────────────────────────────────────────────────────────────

    def update(self, variable: str, value: float, step: Optional[int] = None):
        if variable not in self._vars:
            self._vars[variable] = VarState(
                name=variable, history=deque(maxlen=self._history_len)
            )

        s   = self._vars[variable]
        val = float(value)
        s.history.append(val)
        # deque(maxlen) handles trim automatically — no manual pop(0)

        step = step if step is not None else self._step
        self._step = max(self._step, step + 1)

        if not s._initialized:
            s.ema          = val
            s._initialized = True
        else:
            s.ema = self.beta * s.ema + (1 - self.beta) * val

        old_track = s.track
        s.track   = self._classify(val, s.ema)

        if len(s.history) >= 20:
            s.state = "A" if float(np.var(list(s.history)[-20:])) < 1e-4 else "P"

        if old_track in ("x_M", "x_W") and s.track == "Y_W":
            s.alert      = True
            s.alert_step = step
            self._alerts.append({
                "variable": variable, "step": step,
                "from_track": old_track, "to_track": s.track,
                "value": val, "ema": s.ema,
            })
        elif s.track != "Y_W":
            s.alert = False

    # ── Query ──────────────────────────────────────────────────────────────

    def check(self) -> Dict[str, Dict]:
        return {
            name: {"track": s.track, "state": s.state, "ema": round(s.ema, 6),
                   "alert": s.alert, "alert_step": s.alert_step}
            for name, s in self._vars.items()
        }

    def is_converged(self) -> bool:
        if not self._vars: return False
        return all(s.state == "A" for s in self._vars.values())

    def alerts(self) -> List[Dict]:
        return list(self._alerts)

    def clear_alerts(self):
        self._alerts.clear()
        for s in self._vars.values():
            s.alert = False; s.alert_step = None

    def locate_fault(self, variable: str) -> str:
        relevant = [a for a in self._alerts if a["variable"] == variable]
        if not relevant:
            return f"No anomaly detected for '{variable}'"
        a = relevant[-1]
        return (f"Anomaly in '{variable}' at step {a['step']}: "
                f"track jumped {a['from_track']} -> {a['to_track']} "
                f"(value={a['value']:.4f}, ema={a['ema']:.4f})")

    def render_panels(self):
        print("\n-- NipGraph Monitor " + "-" * 40)
        for name, info in self.check().items():
            alert_str = " !! ALERT"     if info["alert"]          else ""
            conv_str  = " == CONVERGED" if info["state"] == "A"   else ""
            print(f"  {name:<20} track={info['track']}  "
                  f"ema={info['ema']:.5f}  "
                  f"state={info['state']}{alert_str}{conv_str}")
        print("-" * 58 + "\n")

    def status(self) -> Dict:
        return {"step": self._step, "variables": list(self._vars.keys()),
                "converged": self.is_converged(), "alerts": len(self._alerts)}

    def health(self) -> Dict:
        return {"status": "ok", **self.status()}

    def __repr__(self) -> str:
        return (f"NipGraph(step={self._step}, "
                f"variables={list(self._vars.keys())}, "
                f"converged={self.is_converged()}, "
                f"alerts={len(self._alerts)})")

    # ── Internal ───────────────────────────────────────────────────────────

    def _classify(self, val: float, ema: float) -> str:
        band_width = self.band * (abs(ema) + self.absolute_floor)
        positive   = (ema >= 0)
        stable     = (abs(val - ema) <= band_width)
        if positive and stable:     return "x_M"
        if positive and not stable: return "x_W"
        if not positive and stable: return "Y_M"
        return "Y_W"
