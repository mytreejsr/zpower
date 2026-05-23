# zpower/utils/config.py  — v1.3.0
# v1.3: validate() uses ValueError instead of assert (assert can be disabled with -O)
#        validate() covers all fields
#        Added __repr__
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

    # GradShield
    grad_clip_norm:          float = 5.0
    grad_adaptive:           bool  = True
    grad_k:                  float = 2.0
    grad_vanish_factor:      float = 0.001
    grad_explode_k:          float = 6.0

    # StabilityCore
    stability_ema_beta:      float = 0.95
    stability_plateau_steps: int   = 10
    stability_plateau_delta: float = 0.001

    # NipGraph
    nipgraph_band:           float = 0.10
    nipgraph_ema_beta:       float = 0.90
    nipgraph_abs_floor:      float = 0.10

    # Weight system
    vault_threshold:         float = 0.75
    vault_max_per_layer:     int   = 5
    guard_lambda:            float = 0.8
    guard_adapt_rate:        float = 0.1
    fisher_batches:          int   = 10

    # AutoHeal
    heal_lr_factor:          float = 0.5
    heal_patience:           int   = 5
    heal_max_heals:          int   = 5
    heal_min_lr:             float = 1e-7
    heal_strategy:           str   = "both"

    # General
    memory_warn_pct:         float = 0.85

    def validate(self) -> "ZPConfig":
        """Validate all config fields. Raises ValueError on invalid values."""
        # OTUX
        if self.otux_dim <= 0:
            raise ValueError(f"otux_dim must be > 0, got {self.otux_dim}")
        if self.otux_max_entries <= 0:
            raise ValueError(f"otux_max_entries must be > 0, got {self.otux_max_entries}")
        if not (0 <= self.otux_forget_thresh < self.otux_importance_thresh <= 1.0):
            raise ValueError(
                f"Need 0 <= otux_forget_thresh ({self.otux_forget_thresh}) "
                f"< otux_importance_thresh ({self.otux_importance_thresh}) <= 1.0"
            )
        if self.otux_buffer_strikes <= 0:
            raise ValueError(f"otux_buffer_strikes must be > 0, got {self.otux_buffer_strikes}")
        if self.otux_decay <= 0 or self.otux_decay > 1:
            raise ValueError(f"otux_decay must be in (0, 1], got {self.otux_decay}")

        # GradShield
        if self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be > 0, got {self.grad_clip_norm}")
        if self.grad_k <= 0:
            raise ValueError(f"grad_k must be > 0, got {self.grad_k}")
        if self.grad_explode_k <= 0:
            raise ValueError(f"grad_explode_k must be > 0, got {self.grad_explode_k}")

        # StabilityCore
        if not (0 < self.stability_ema_beta < 1):
            raise ValueError(f"stability_ema_beta must be in (0, 1), got {self.stability_ema_beta}")
        if self.stability_plateau_steps <= 0:
            raise ValueError(f"stability_plateau_steps must be > 0, got {self.stability_plateau_steps}")

        # Weight system
        if not (0 < self.vault_threshold <= 1.0):
            raise ValueError(f"vault_threshold must be in (0, 1], got {self.vault_threshold}")
        if self.vault_max_per_layer <= 0:
            raise ValueError(f"vault_max_per_layer must be > 0, got {self.vault_max_per_layer}")
        if self.guard_lambda < 0:
            raise ValueError(f"guard_lambda must be >= 0, got {self.guard_lambda}")
        if not (0 <= self.guard_adapt_rate <= 1):
            raise ValueError(f"guard_adapt_rate must be in [0, 1], got {self.guard_adapt_rate}")

        # AutoHeal
        if self.heal_lr_factor <= 0 or self.heal_lr_factor >= 1:
            raise ValueError(f"heal_lr_factor must be in (0, 1), got {self.heal_lr_factor}")
        if self.heal_patience <= 0:
            raise ValueError(f"heal_patience must be > 0, got {self.heal_patience}")
        if self.heal_max_heals <= 0:
            raise ValueError(f"heal_max_heals must be > 0, got {self.heal_max_heals}")
        if self.heal_min_lr <= 0:
            raise ValueError(f"heal_min_lr must be > 0, got {self.heal_min_lr}")
        valid_strategies = ("both", "rollback_only", "lr_only", "restart")
        if self.heal_strategy not in valid_strategies:
            raise ValueError(f"heal_strategy must be one of {valid_strategies}, got {self.heal_strategy}")

        # General
        if not (0 < self.memory_warn_pct <= 1):
            raise ValueError(f"memory_warn_pct must be in (0, 1], got {self.memory_warn_pct}")

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

    def __repr__(self) -> str:
        return (f"ZPConfig(dim={self.otux_dim}, adaptive={self.grad_adaptive}, "
                f"heal_strategy={self.heal_strategy})")


DEFAULT_CONFIG = ZPConfig()
