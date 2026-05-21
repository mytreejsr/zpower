# zpower/heal.py  —  AutoHeal Engine v1.1
# Lightweight, in-memory training recovery. No external DB, no heavy deps.
#
# Logic:
#   On NaN loss OR severe divergence (StabilityCore state = "diverging")
#   OR GradShield state = EXPLODING for N consecutive steps:
#     1. Pause training loop
#     2. Roll back model weights to last WeightVault snapshot
#     3. Reduce optimizer LR by heal_lr_factor
#     4. Resume training
#
# Usage:
#   healer = AutoHeal(model, vault, optimizer)
#   ... in training loop ...
#   action = healer.on_step(loss_value, grad_state, stability_info)
#   if action == "skip":   continue
#   if action == "healed": print("recovered!")
from __future__ import annotations

import math
from typing import Any, Dict, Optional

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


class AutoHeal:
    """
    AutoHeal — In-memory training failure recovery.

    Detects three failure modes:
      1. NaN / Inf loss           (immediate trigger)
      2. Diverging loss           (StabilityCore state == 'diverging')
      3. Consecutive explosions   (GradShield EXPLODING for N steps)

    Recovery steps (in order):
      1. Pause: skip current batch
      2. Rollback: restore WeightVault best snapshot to model
      3. LR reduction: optimizer LR *= heal_lr_factor
      4. Resume: return 'healed' so caller can log and continue

    Parameters
    ----------
    model            : ZPowerModel or raw torch model
    vault            : WeightVault instance
    optimizer        : torch optimizer
    heal_lr_factor   : LR multiplier on heal event (default 0.5)
    explode_patience : Consecutive EXPLODING steps before triggering heal
    max_heals        : Max number of heal events before raising (default 5)
    min_lr           : Floor for LR — never reduce below this (default 1e-7)
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
    ):
        self._model     = model
        self._vault     = vault
        self._optimizer = optimizer
        self.lr_factor  = heal_lr_factor
        self.patience   = explode_patience
        self.max_heals  = max_heals
        self.min_lr     = min_lr

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
        return self._heal(
            reason="diverging"        if diverging          else
                   "explode_patience" if consecutive_explode else
                   "nan"
        )

    # ── Status ─────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        return {
            "heal_count":      self._heal_count,
            "max_heals":       self.max_heals,
            "explode_streak":  self._explode_streak,
            "heal_log":        list(self._heal_log),
        }

    def health(self) -> Dict:
        return {"status": "ok", **self.status()}

    # ── Internal ───────────────────────────────────────────────────────────

    def _heal(self, reason: str) -> str:
        if self._heal_count >= self.max_heals:
            raise RuntimeError(
                f"[AutoHeal] Max heal events ({self.max_heals}) reached. "
                f"Training cannot self-recover. Last reason: {reason}. "
                f"Consider reducing LR or checking your data."
            )

        rollback_ok = self._rollback_weights()
        lr_applied  = self._reduce_lr()
        self._explode_streak = 0
        self._heal_count    += 1

        event = {
            "heal_n":      self._heal_count,
            "reason":      reason,
            "rollback":    rollback_ok,
            "new_lr":      lr_applied,
        }
        self._heal_log.append(event)

        print(
            f"[AutoHeal] Heal #{self._heal_count} triggered. "
            f"Reason: {reason}. "
            f"Rollback: {'✓' if rollback_ok else '✗ (no vault snapshot)'}. "
            f"New LR: {lr_applied:.2e}"
        )
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
            print(f"[AutoHeal] Rollback failed: {e}")
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
