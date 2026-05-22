# zpower/utils/config.py  — v1.2.0
# New: GradShield adaptive params, AutoHeal params, nipgraph_abs_floor, to_json()
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class ZPConfig:
    # OTUX-S
    otux_dim:                int   = 256
    otux_max_entries:        int   = 10_000
    otux_importance_thresh:  float = 0.65
    otux_forget_thresh:      float = 0.30
    otux_buffer_strikes:     int   = 3
    otux_context_window:     int   = 16
    otux_decay:              float = 0.95

    # GradShield — v1.2: adaptive params
    grad_clip_norm:          float = 5.0
    grad_adaptive:           bool  = True
    grad_k:                  float = 2.0
    grad_vanish_factor:      float = 0.001
    grad_explode_k:          float = 6.0

    # StabilityCore
    stability_ema_beta:      float = 0.95
    stability_plateau_steps: int   = 10
    stability_plateau_delta: float = 0.001

    # NipGraph — v1.2: absolute_floor
    nipgraph_band:           float = 0.10
    nipgraph_ema_beta:       float = 0.90
    nipgraph_abs_floor:      float = 0.10

    # Weight system
    vault_threshold:         float = 0.75
    vault_max_per_layer:     int   = 5
    guard_lambda:            float = 0.8
    guard_adapt_rate:        float = 0.1
    fisher_batches:          int   = 10

    # AutoHeal — v1.2
    heal_lr_factor:          float = 0.5
    heal_patience:           int   = 5
    heal_max_heals:          int   = 5
    heal_min_lr:             float = 1e-7

    # General
    memory_warn_pct:         float = 0.85

    def validate(self):
        assert 0 < self.otux_forget_thresh < self.otux_importance_thresh <= 1.0
        assert 0 < self.vault_threshold <= 1.0
        assert self.grad_clip_norm > 0
        assert 0 < self.stability_ema_beta < 1
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ZPConfig":
        with open(path) as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid).validate()


DEFAULT_CONFIG = ZPConfig()
