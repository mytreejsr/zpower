# zpower/monitor/nipgraph.py  —  NipGraph v1.2.0
# v1.2 fixes:
#   HIGH: step counter drift — use max() not increment
#   HIGH: near-zero EMA misclassification — absolute_floor parameter
#   FIX:  _initialized flag replaces EMA == 0.0 sentinel in VarState
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
    _initialized: bool = False   # v1.2: replaces ema==0.0 sentinel
    alert:      bool  = False
    alert_step: Optional[int] = None
    history:    List[float]   = field(default_factory=list)


class NipGraph:
    """
    NipGraph v1.2.0 — Parity-aware training anomaly detection.

    v1.2 fixes:
      step counter drift — self._step = max(self._step, step+1),
        prevents double-counting when GradShield + StabilityCore
        both call update() at the same training step.

      near-zero EMA false alerts — absolute_floor (default 0.1):
        band_width = band * (abs(ema) + absolute_floor)
        Prevents band from collapsing to near-zero on converged models.

      _initialized flag — safe when first observation value == 0.0.
    """

    def __init__(
        self,
        variables:      List[str],
        band_width:     float = 0.10,
        ema_beta:       float = 0.90,
        history_len:    int   = 200,
        absolute_floor: float = 0.10,   # v1.2: prevents near-zero band collapse
    ):
        self.band           = band_width
        self.beta           = ema_beta
        self.absolute_floor = absolute_floor
        self._vars: Dict[str, VarState] = {v: VarState(name=v) for v in variables}
        self._history_len   = history_len
        self._step          = 0
        self._alerts: List[Dict] = []

    # ── Update ─────────────────────────────────────────────────────────────

    def update(self, variable: str, value: float, step: Optional[int] = None):
        if variable not in self._vars:
            self._vars[variable] = VarState(name=variable)

        s   = self._vars[variable]
        val = float(value)
        s.history.append(val)
        if len(s.history) > self._history_len:
            s.history.pop(0)

        step = step if step is not None else self._step
        # v1.2: use max() to prevent counter drift
        self._step = max(self._step, step + 1)

        # v1.2: _initialized flag — safe when first value is 0.0
        if not s._initialized:
            s.ema          = val
            s._initialized = True
        else:
            s.ema = self.beta * s.ema + (1 - self.beta) * val

        old_track = s.track
        s.track   = self._classify(val, s.ema)

        if len(s.history) >= 20:
            s.state = "A" if float(np.var(s.history[-20:])) < 1e-4 else "P"

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
                f"track jumped {a['from_track']} → {a['to_track']} "
                f"(value={a['value']:.4f}, ema={a['ema']:.4f})")

    def render_panels(self):
        print("\n── NipGraph Monitor ─────────────────────────────────")
        for name, info in self.check().items():
            alert_str = " ⚠ ALERT"     if info["alert"]          else ""
            conv_str  = " ✓ CONVERGED" if info["state"] == "A"   else ""
            print(f"  {name:<20} track={info['track']}  "
                  f"ema={info['ema']:.5f}  "
                  f"state={info['state']}{alert_str}{conv_str}")
        print("─────────────────────────────────────────────────────\n")

    def status(self) -> Dict:
        return {"step": self._step, "variables": list(self._vars.keys()),
                "converged": self.is_converged(), "alerts": len(self._alerts)}

    def health(self) -> Dict:
        return {"status": "ok", **self.status()}

    # ── Internal ───────────────────────────────────────────────────────────

    def _classify(self, val: float, ema: float) -> str:
        # v1.2: absolute_floor prevents band collapsing near zero
        band_width = self.band * (abs(ema) + self.absolute_floor)
        positive   = (ema >= 0)
        stable     = (abs(val - ema) <= band_width)
        if positive and stable:     return "x_M"
        if positive and not stable: return "x_W"
        if not positive and stable: return "Y_M"
        return "Y_W"
