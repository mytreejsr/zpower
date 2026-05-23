# zpower/heal.py  —  AutoHeal Engine v1.3.0
# Lightweight, in-memory training recovery. No external DB, no heavy deps.
#
# v1.3 changes:
#   - Multiple heal strategies: 'both', 'rollback_only', 'lr_only', 'restart'
#   - Replaced print() with zplog for structured logging
#   - Added reset() method for reuse between training runs
#   - Added __repr__
#
# Logic:
#   On NaN loss OR severe divergence (StabilityCore state = "diverging")
#   OR GradShield state = EXPLODING for N consecutive steps:
#     Strategy 'both':          rollback weights + reduce LR (v1.1/v1.2 default)
#     Strategy 'rollback_only': rollback weights only, keep LR
#     Strategy 'lr_only':       reduce LR only, no rollback
#     Strategy 'restart':       rollback + reduce LR + reset optimizer momentum
from __future__ import annotations

import math
from typing import Any, Dict, Optional

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

from zpower.utils import logging as zplog

_VALID_STRATEGIES = ("both", "rollback_only", "lr_only", "restart")


class AutoHeal:
    """
    AutoHeal v1.3.0 — In-memory training failure recovery.

    Detects three failure modes:
      1. NaN / Inf loss           (immediate trigger)
      2. Diverging loss           (StabilityCore state == 'diverging')
      3. Consecutive explosions   (GradShield EXPLODING for N steps)

    Recovery strategies (controlled by `strategy` parameter):
      'both'          : Rollback weights + reduce LR (default, v1.1/v1.2 behaviour)
      'rollback_only' : Rollback weights only, keep current LR
      'lr_only'       : Reduce LR only, no weight rollback
      'restart'       : Rollback + reduce LR + reset optimizer momentum

    Parameters
    ----------
    model            : ZPowerModel or raw torch model
    vault            : WeightVault instance
    optimizer        : torch optimizer
    heal_lr_factor   : LR multiplier on heal event (default 0.5)
    explode_patience : Consecutive EXPLODING steps before triggering heal
    max_heals        : Max number of heal events before raising (default 5)
    min_lr           : Floor for LR — never reduce below this (default 1e-7)
    strategy         : Recovery strategy (default 'both')
    """

    HEALED   = "healed"
    SKIP     = "skip"
    CONTINUE = "continue"

    def __init__(
        self,
        model,
        vault,
        optimizer,
        heal_lr_factor:   float = 0.5,
        explode_patience: int   = 5,
        max_heals:        int   = 5,
        min_lr:           float = 1e-7,
        strategy:         str   = "both",
    ):
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"AutoHeal: invalid strategy='{strategy}'. "
                f"Valid options: {_VALID_STRATEGIES}"
            )
        self._model     = model
        self._vault     = vault
        self._optimizer = optimizer
        self.lr_factor  = heal_lr_factor
        self.patience   = explode_patience
        self.max_heals  = max_heals
        self.min_lr     = min_lr
        self.strategy   = strategy

        self._explode_streak = 0
        self._heal_count     = 0
        self._heal_log: list = []

    # ── Main hook — call once per training step ────────────────────────────

    def on_step(
        self,
        loss_value:     float,
        grad_state:     str  = "healthy",
        stability_info: Optional[Dict] = None,
    ) -> str:
        """
        Evaluate current training step health.

        Parameters
        ----------
        loss_value     : float — current batch loss (loss.item())
        grad_state     : str   — GradShield.last_state() or 'healthy'
        stability_info : dict  — StabilityCore.update() return value (optional)

        Returns
        -------
        'continue' → step is healthy, proceed normally
        'skip'     → bad step detected, skip backward/optimizer (caller must continue)
        'healed'   → heal was applied, caller should log and continue next epoch
        """
        # ── Failure detection ─────────────────────────────────────────────
        nan_detected       = not math.isfinite(loss_value)
        diverging          = (stability_info or {}).get("state") == "diverging"
        exploding_grad     = (grad_state == "exploding")

        if exploding_grad:
            self._explode_streak += 1
        else:
            self._explode_streak = 0

        consecutive_explode = self._explode_streak >= self.patience

        if not (nan_detected or diverging or consecutive_explode):
            return self.CONTINUE

        # ── Skip current batch if NaN (don't backward on NaN) ────────────
        if nan_detected:
            return self.SKIP

        # ── Trigger heal ──────────────────────────────────────────────────
        reason = ("diverging"        if diverging          else
                  "explode_patience" if consecutive_explode else
                  "nan")
        return self._heal(reason=reason)

    # ── Status ─────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        return {
            "heal_count":      self._heal_count,
            "max_heals":       self.max_heals,
            "explode_streak":  self._explode_streak,
            "strategy":        self.strategy,
            "heal_log":        list(self._heal_log),
        }

    def health(self) -> Dict:
        return {"status": "ok", **self.status()}

    def reset(self):
        """Reset heal state for reuse in a new training run."""
        self._explode_streak = 0
        self._heal_count     = 0
        self._heal_log.clear()

    def __repr__(self) -> str:
        return (f"AutoHeal(heals={self._heal_count}/{self.max_heals}, "
                f"strategy={self.strategy}, "
                f"streak={self._explode_streak}/{self.patience})")

    # ── Internal ───────────────────────────────────────────────────────────

    def _heal(self, reason: str) -> str:
        if self._heal_count >= self.max_heals:
            raise RuntimeError(
                f"[AutoHeal] Max heal events ({self.max_heals}) reached. "
                f"Training cannot self-recover. Last reason: {reason}. "
                f"Consider reducing LR or checking your data."
            )

        rollback_ok = False
        lr_applied  = 0.0

        if self.strategy in ("both", "rollback_only", "restart"):
            rollback_ok = self._rollback_weights()

        if self.strategy in ("both", "lr_only", "restart"):
            lr_applied = self._reduce_lr()

        if self.strategy == "restart":
            self._reset_optimizer_momentum()

        self._explode_streak = 0
        self._heal_count    += 1

        event = {
            "heal_n":      self._heal_count,
            "reason":      reason,
            "rollback":    rollback_ok,
            "new_lr":      lr_applied,
            "strategy":    self.strategy,
        }
        self._heal_log.append(event)

        zplog.warning("AutoHeal",
            f"Heal #{self._heal_count} triggered. "
            f"Reason: {reason}. Strategy: {self.strategy}. "
            f"Rollback: {'yes' if rollback_ok else 'no (no vault snapshot)'}. "
            f"New LR: {lr_applied:.2e}")
        return self.HEALED

    def _rollback_weights(self) -> bool:
        """Restore best WeightVault snapshot to model. Returns True if successful."""
        if not _TORCH:
            return False
        if self._vault is None:
            return False

        try:
            best_sd = self._vault.get_best_state_dict()
            if not best_sd:
                return False

            # Determine target model (unwrap ZPowerModel if needed)
            target = self._model
            if hasattr(target, "original_model"):
                target = target.original_model

            if hasattr(target, "load_state_dict"):
                # Only load keys that exist in current model
                current_sd = target.state_dict()
                filtered   = {k: v for k, v in best_sd.items() if k in current_sd}
                if filtered:
                    current_sd.update(filtered)
                    target.load_state_dict(current_sd)
                    return True
        except Exception as e:
            zplog.error("AutoHeal", f"Rollback failed: {e}")
        return False

    def _reduce_lr(self) -> float:
        """Multiply all optimizer LR groups by lr_factor. Returns new LR."""
        if self._optimizer is None:
            return 0.0
        try:
            for group in self._optimizer.param_groups:
                new_lr = max(group["lr"] * self.lr_factor, self.min_lr)
                group["lr"] = new_lr
            return float(self._optimizer.param_groups[0]["lr"])
        except Exception:
            return 0.0

    def _reset_optimizer_momentum(self) -> None:
        """Reset optimizer momentum/buffers for a fresh start after rollback."""
        if self._optimizer is None:
            return
        try:
            for group in self._optimizer.param_groups:
                for p in group.get("params", []):
                    state = self._optimizer.state.get(p, {})
                    for key in ("momentum_buffer", "exp_avg", "exp_avg_sq",
                                "max_exp_avg_sq", "step"):
                        if key in state:
                            del state[key]
        except Exception:
            pass
