# zpower/stabilize/grad_shield.py  —  GradShield v1.2.0
# v1.2 fixes:
#   PERF: O(1) health_rate via running counters (was O(N) scan)
#   STRUCT: __del__ + context manager for automatic hook cleanup
#   API: consistent health() method
import math
from collections import deque
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

_WARMUP_STEPS = 20


class GradShield:
    """
    GradShield v1.2.0 — Adaptive gradient health monitor.

    v1.1: adaptive per-layer thresholds, deque history
    v1.2: O(1) health_rate (running counters), __del__/context manager,
          health() API
    """

    VANISHING  = "vanishing"
    HEALTHY    = "healthy"
    WARNING    = "warning"
    EXPLODING  = "exploding"

    def __init__(
        self,
        clip_norm:      float = 5.0,
        vanish_thresh:  float = 1e-7,
        explode_thresh: float = 50.0,
        adaptive:       bool  = True,
        k:              float = 2.0,
        vanish_factor:  float = 0.001,
        explode_k:      float = 6.0,
        nipgraph        = None,
    ):
        self._fixed_clip    = clip_norm
        self._fixed_vanish  = vanish_thresh
        self._fixed_explode = explode_thresh
        self.adaptive       = adaptive
        self.k              = k
        self.vanish_factor  = vanish_factor
        self.explode_k      = explode_k
        self._nipgraph      = nipgraph

        self._history: deque = deque(maxlen=500)
        self._step    = 0
        self._hooks:  List   = []
        self._layer_stats: Dict[str, Dict] = {}

        # v1.2: O(1) running counters for health_rate
        self._window_total   = 0
        self._window_healthy = 0

    # ── Core API ───────────────────────────────────────────────────────────

    def check(self, gradients, layer_name: str = "_global") -> str:
        norm = self._norm(gradients)
        clip_n, vanish_t, explode_t = self._thresholds(layer_name, norm)
        if norm < vanish_t:    return self.VANISHING
        if norm <= clip_n:     return self.HEALTHY
        if norm <= explode_t:  return self.WARNING
        return self.EXPLODING

    def shield(self, gradients, layer_name: str = "_global"):
        norm  = self._norm(gradients)
        self._update_layer_stats(layer_name, norm)
        state = self.check(gradients, layer_name)
        clip_n, _, _ = self._thresholds(layer_name, norm)

        # v1.2: update running counters
        if len(self._history) == self._history.maxlen:
            oldest = self._history[0]
            self._window_total   -= 1
            if oldest["state"] == self.HEALTHY:
                self._window_healthy -= 1

        self._history.append({"step": self._step, "norm": norm,
                               "state": state, "layer": layer_name})
        self._window_total   += 1
        if state == self.HEALTHY:
            self._window_healthy += 1
        self._step += 1

        if self._nipgraph is not None:
            self._nipgraph.update("grad_norm", norm, step=self._step)

        if state in (self.WARNING, self.EXPLODING):
            gradients = self._clip(gradients, clip_n)
        return gradients

    def attach_to_model(self, model) -> "GradShield":
        if not _TORCH:
            raise ImportError("GradShield.attach_to_model() requires torch")
        self._remove_hooks()
        for name, param in model.named_parameters():
            if param.requires_grad:
                h = param.register_hook(
                    lambda grad, n=name: self._hook_fn(grad, n)
                )
                self._hooks.append(h)
        return self

    def detach(self):
        self._remove_hooks()

    # ── Context manager support ────────────────────────────────────────────

    def __enter__(self): return self
    def __exit__(self, *_): self.detach()
    def __del__(self):
        try: self._remove_hooks()
        except Exception: pass

    # ── Diagnostics ────────────────────────────────────────────────────────

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def last_state(self) -> Optional[str]:
        return self._history[-1]["state"] if self._history else None

    def layer_stats(self) -> Dict[str, Dict]:
        out = {}
        for name, s in self._layer_stats.items():
            mn  = s["mean"]
            std = s["std"]
            out[name] = {
                "mean":          round(mn, 6),
                "std":           round(std, 6),
                "adaptive_clip": round(mn + self.k * std, 6),
                "samples":       len(s["norms"]),
            }
        return out

    def status(self) -> Dict:
        # v1.2: O(1) health_rate from running counters
        hr = round(self._window_healthy / max(self._window_total, 1), 3)
        return {
            "status":         "ok",
            "steps":          self._step,
            "health_rate":    hr,
            "last_state":     self.last_state(),
            "hooks_active":   len(self._hooks),
            "adaptive":       self.adaptive,
            "layers_tracked": len(self._layer_stats),
        }

    def health(self) -> Dict:
        return {"status": "ok", **self.status()}

    # ── Internal ───────────────────────────────────────────────────────────

    def _thresholds(self, layer_name, norm):
        if not self.adaptive:
            return self._fixed_clip, self._fixed_vanish, self._fixed_explode
        s = self._layer_stats.get(layer_name, {})
        if len(s.get("norms", [])) < _WARMUP_STEPS:
            return self._fixed_clip, self._fixed_vanish, self._fixed_explode
        mean = s["mean"]; std = s["std"]
        clip_n    = max(mean + self.k * std, 0.01)
        vanish_t  = max(mean * self.vanish_factor if mean > 0 else self._fixed_vanish, 1e-12)
        explode_t = max(mean + self.explode_k * std, clip_n * 2.0)
        return clip_n, vanish_t, explode_t

    def _update_layer_stats(self, layer_name, norm):
        if layer_name not in self._layer_stats:
            self._layer_stats[layer_name] = {
                "norms": deque(maxlen=200), "mean": 0.0, "std": 1.0,
                "_m2": 0.0, "_count": 0,
            }
        s = self._layer_stats[layer_name]
        s["norms"].append(norm)
        s["_count"] += 1
        delta      = norm - s["mean"]
        s["mean"] += delta / s["_count"]
        delta2     = norm - s["mean"]
        s["_m2"]  += delta * delta2
        if s["_count"] >= 2:
            s["std"] = max(math.sqrt(s["_m2"] / (s["_count"] - 1)), 1e-8)

    def _hook_fn(self, grad, layer_name):
        if grad is not None:
            return self.shield(grad, layer_name)
        return grad

    def _norm(self, g) -> float:
        if _TORCH and isinstance(g, torch.Tensor):
            return float(g.norm().item())
        return float(np.linalg.norm(np.asarray(g, dtype=np.float32)))

    def _clip(self, g, max_norm):
        norm = self._norm(g)
        if norm == 0: return g
        scale = min(1.0, max_norm / norm)
        if _TORCH and isinstance(g, torch.Tensor): return g * scale
        return np.asarray(g, dtype=np.float32) * scale

    def _remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()
