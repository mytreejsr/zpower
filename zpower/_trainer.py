# zpower/_trainer.py  —  Trainer v1.3.0
# v1.3 fixes:
#   CRITICAL: self._ema_last(model) → _ema_last(model) (was module-level function, not method)
#   HEAL_STRATEGIES: support 'both', 'rollback_only', 'lr_only', 'restart'
#   Uses zpower.utils.logging instead of bare print()
#   Dead code removed
from __future__ import annotations
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

from zpower.utils import logging as zplog

_VALID_HEAL_STRATEGIES = {"both", "rollback_only", "lr_only", "restart"}


def _run_fit(trainer_obj, dataset, epochs: int, lr: float, **kwargs):
    if not _TORCH:
        raise ImportError("zp.Trainer requires torch")

    model     = trainer_obj.model
    loss_fn   = kwargs.get("loss_fn", nn.CrossEntropyLoss())
    optimizer = kwargs.get("optimizer", None)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    device = kwargs.get("device", "cpu")
    if torch.cuda.is_available() and device == "cpu":
        device = "cuda"

    # v1.3: validate heal strategy
    heal_strategy = kwargs.get("heal_strategy", "both")
    if heal_strategy not in _VALID_HEAL_STRATEGIES:
        raise ValueError(
            f"Invalid heal_strategy='{heal_strategy}'. "
            f"Valid options: {sorted(_VALID_HEAL_STRATEGIES)}"
        )

    model.to(device)
    model.train()

    # AutoHeal setup
    healer = None
    _auto  = getattr(model, "_auto_heal", False) or kwargs.get("auto_heal", False)
    if _auto:
        from zpower.heal import AutoHeal
        vault  = getattr(model, "_vault", None)
        healer = AutoHeal(
            model            = model,
            vault            = vault,
            optimizer        = optimizer,
            heal_lr_factor   = kwargs.get("heal_lr_factor", 0.5),
            explode_patience = kwargs.get("heal_patience", 5),
            max_heals        = kwargs.get("max_heals", 5),
            strategy         = heal_strategy,
        )

    history     = []
    heal_events = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0
        zp_info: Dict = {}

        for batch in dataset:
            optimizer.zero_grad()

            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            if hasattr(inputs, "to"):  inputs  = inputs.to(device)
            if targets is not None and hasattr(targets, "to"):
                targets = targets.to(device)

            outputs  = model(inputs)
            loss     = loss_fn(outputs, targets) if targets is not None else outputs.mean()
            loss_val = float(loss.item()) if torch.isfinite(loss) else float("nan")

            # ── Get REAL grad state (not always "healthy") ──────────────
            gs_state  = "healthy"
            stab_info = {}
            if hasattr(model, "_stabilizer") and model._stabilizer:
                actual = model._stabilizer.grad_shield.last_state()
                if actual:
                    gs_state = actual
                if hasattr(model._stabilizer, "stability_core"):
                    # v1.3 FIX: _ema_last is a module-level function, NOT self.method
                    fallback = _ema_last(model)
                    stab_info = model._stabilizer.stability_core.update(
                        loss_val if torch.isfinite(loss) else fallback
                    )

            # AutoHeal check
            if healer is not None:
                action = healer.on_step(loss_val, gs_state, stab_info)
                if action == "skip":
                    heal_events += 1
                    continue
                if action == "healed":
                    heal_events += 1
                    continue

            if hasattr(model, "zp_on_loss") and torch.isfinite(loss):
                zp_info = model.zp_on_loss(loss_val, optimizer)

            if hasattr(model, "ewc_penalty"):
                penalty = model.ewc_penalty()
                if isinstance(penalty, torch.Tensor) and penalty.item() > 0:
                    loss = loss + penalty

            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
            else:
                zplog.warning("Trainer",
                    f"Non-finite loss at epoch {epoch}. "
                    "Enable auto_heal=True to recover automatically.")
                continue

            # Pass REAL grad_health to vault
            if hasattr(model, "zp_on_step_end"):
                model.zp_on_step_end({
                    "loss":        loss_val,
                    "grad_health": gs_state,
                    "curvature":   zp_info.get("curvature", "moderate"),
                    "val_accuracy": kwargs.get("val_accuracy", 0.5),
                    "confidence":   kwargs.get("confidence",  0.5),
                })

            epoch_loss += loss_val
            n_batches  += 1

        avg_loss  = epoch_loss / max(n_batches, 1)
        heal_str  = f"  heals={heal_events}" if heal_events > 0 else ""
        history.append({"epoch": epoch, "loss": avg_loss, "heals": heal_events})

        zplog.info("Trainer",
            f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}"
            f"  curvature={zp_info.get('curvature','?')}"
            f"  lr_signal={zp_info.get('lr_signal',1.0):.3f}"
            f"{heal_str}")

        if hasattr(model, "monitor") and model.monitor is not None:
            if model.monitor.is_converged():
                zplog.info("Trainer", f"NipGraph: converged at epoch {epoch}")
                break

    return history


def _ema_last(model) -> float:
    """Safe fallback: last known EMA from StabilityCore."""
    try:
        return model._stabilizer.stability_core._ema
    except Exception:
        return 1.0
