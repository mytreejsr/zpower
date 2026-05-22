# zpower/__init__.py  —  ZPower v1: Intelligence layer for AI/ML
# Author: NNN Bhoi
# import zpower as zp

from __future__ import annotations
from typing import Any, Optional

__version__ = "1.2.1"
__author__  = "NNN Bhoi"

# ── Submodule exports ──────────────────────────────────────────────────────
from zpower.memory.otux              import OtuxStore, ImportanceWeights
from zpower.stabilize.grad_shield    import GradShield
from zpower.stabilize.stability_core import StabilityCore
from zpower.stabilize.model_stabilizer import ModelStabilizer
from zpower.monitor.nipgraph         import NipGraph
from zpower.weights.vault            import WeightVault
from zpower.weights.surgeon          import WeightSurgeon
from zpower.weights.guard            import WeightGuard
from zpower.compute.safe_math        import SafeMath, safe_loss, safe_divide
from zpower.heal                      import AutoHeal

# Subpackage namespaces
from zpower import memory, stabilize, monitor, weights, compute, compat, utils

# ── zp.attach() ────────────────────────────────────────────────────────────
def attach(
    model: Any,
    *,
    memory:       str  = "otux_selective",
    stabilize:    bool = True,
    monitor:      bool = True,
    weight_vault: bool = False,
    weight_guard: bool = False,
    auto_heal:    bool = False,
    otux_dim:     int  = 256,
):
    """
    Augment any existing model with ZPower intelligence.

    Works with PyTorch nn.Module, HuggingFace PreTrainedModel,
    or any callable with a forward() method.

    The original model is NOT changed — ZPower runs as a side-channel.

    Parameters
    ----------
    model        : Any torch model
    memory       : 'otux_selective' | 'otux_full' | 'none'
    stabilize    : Attach GradShield + StabilityCore to backward pass
    monitor      : Attach NipGraph parity tracking
    weight_vault : Record high-performance weight snapshots
    weight_guard : EWC penalty to protect good weights (requires weight_vault=True)
    auto_heal    : Reserved for future self-healing (v2)
    otux_dim     : OTUX-S vector dimension

    Returns
    -------
    ZPowerModel  — behaves exactly like original model

    Example
    -------
    >>> import zpower as zp
    >>> zp_model = zp.attach(my_torch_model, stabilize=True, monitor=True)
    >>> output = zp_model(input_tensor)          # unchanged forward pass
    >>> loss.backward()                          # GradShield active here
    """
    try:
        from zpower.compat.torch_wrap import ZPowerModel
        return ZPowerModel(
            model,
            memory       = memory,
            stabilize    = stabilize,
            monitor      = monitor,
            weight_vault = weight_vault,
            weight_guard = weight_guard,
            auto_heal    = auto_heal,
            otux_dim     = otux_dim,
        )
    except ImportError as e:
        raise ImportError(
            "zp.attach() requires torch. Install with: pip install zpower[torch]"
        ) from e


# ── zp.Trainer ─────────────────────────────────────────────────────────────
class Trainer:
    """
    Drop-in intelligent training loop with full ZPower stack active.

    Automatically wraps model via zp.attach() if not already wrapped.

    Parameters
    ----------
    model          : PyTorch model (raw or already wrapped)
    stabilize      : GradShield + StabilityCore
    monitor        : NipGraph parity tracking
    weight_vault   : Performance-gated weight snapshots
    weight_guard   : EWC catastrophic forgetting prevention
    vault_threshold: Minimum P_score to store a snapshot (default 0.75)
    auto_heal      : AutoHeal engine — auto-recovers from NaN/divergence via vault rollback + LR cut

    Example
    -------
    >>> trainer = zp.Trainer(model, stabilize=True, weight_vault=True, auto_heal=True)
    >>> trainer.fit(train_loader, epochs=20)
    >>> print(trainer.weight_report())
    """

    def __init__(
        self,
        model,
        *,
        stabilize:       bool  = True,
        monitor:         bool  = True,
        weight_vault:    bool  = True,
        weight_guard:    bool  = False,
        vault_threshold: float = 0.75,
        auto_heal:       bool  = False,
    ):
        from zpower.compat.torch_wrap import ZPowerModel

        if not isinstance(model, ZPowerModel):
            model = attach(
                model,
                stabilize    = stabilize,
                monitor      = monitor,
                weight_vault = weight_vault,
                weight_guard = weight_guard,
                auto_heal    = auto_heal,
            )
            if model.vault:
                model.vault.threshold = vault_threshold

        self.model           = model
        self._vault_threshold = vault_threshold
        self._history: list  = []

    def fit(self, dataset, epochs: int = 10, lr: float = 1e-3, **kwargs):
        """
        Run training.

        Parameters
        ----------
        dataset  : PyTorch DataLoader or any iterable of (input, target) batches
        epochs   : Number of training epochs
        lr       : Learning rate (used only if optimizer not provided in kwargs)
        **kwargs : loss_fn, optimizer, device

        Returns
        -------
        list of {epoch, loss} dicts
        """
        from zpower._trainer import _run_fit
        self._history = _run_fit(self, dataset, epochs=epochs, lr=lr, **kwargs)
        return self._history

    def weight_report(self) -> dict:
        """Summary of WeightVault snapshots and WeightGuard protection."""
        if self.model.vault:
            return self.model.vault.summary()
        return {"message": "weight_vault not enabled — set weight_vault=True"}

    def nipgraph_status(self) -> dict:
        """Current NipGraph monitor state."""
        if self.model.monitor:
            return self.model.monitor.check()
        return {"message": "monitor not enabled"}

    def zp_status(self) -> dict:
        """Full ZPower component status."""
        return self.model.zp_status()

    def render_monitor(self):
        """Print ASCII NipGraph dashboard."""
        if self.model.monitor:
            self.model.monitor.render_panels()


# ── zp.info() ──────────────────────────────────────────────────────────────
def info():
    """Print ZPower version, modules, and quick usage."""
    try:
        import torch
        torch_ver = torch.__version__
    except ImportError:
        torch_ver = "not installed"

    try:
        import transformers
        hf_ver = transformers.__version__
    except ImportError:
        hf_ver = "not installed"

    print(f"""
╔══════════════════════════════════════════════════════╗
║           ZPower v{__version__} — by {__author__}           ║
║     Intelligence layer for AI/ML systems             ║
╚══════════════════════════════════════════════════════╝

  Environment:
    torch        : {torch_ver}
    transformers : {hf_ver}

  Core Modules:
    zp.memory.OtuxStore      Selective context-aware memory (OTUX-S)
    zp.stabilize             GradShield + StabilityCore + ModelStabilizer
    zp.weights               WeightVault + WeightSurgeon + WeightGuard
    zp.monitor.NipGraph      Parity-aware anomaly detection
    zp.compute.SafeMath      NaN-safe math + rational tokenization
    zp.heal.AutoHeal         Auto training recovery (rollback + LR cut)
    zp.compat.augment()      HuggingFace model augmentation
    zp.utils.ZPConfig        Centralized config (save/load JSON)
    zp.utils.logging         Structured logging with level control
    zp.weights.fisher        Fisher Information for weight importance

  Quick start (pre-trained model):
    import zpower as zp
    zp_model = zp.attach(my_model, stabilize=True, weight_vault=True)
    output   = zp_model(input)

  Quick start (training from scratch):
    trainer = zp.Trainer(my_model, stabilize=True, weight_vault=True, auto_heal=True)
    trainer.fit(train_loader, epochs=20)
    print(trainer.weight_report())
    print(trainer.model.health_report())

  Quick start (HuggingFace):
    from transformers import AutoModel
    model    = AutoModel.from_pretrained("bert-base-uncased")
    zp_model = zp.compat.augment(model, stabilize=True, weight_vault=True)
    output   = zp_model(input_ids)

  Install options:
    pip install zpower              # numpy only
    pip install zpower[torch]       # + PyTorch support
    pip install zpower[hf]          # + HuggingFace Transformers
    pip install zpower[full]        # everything
""")


__all__ = [
    # Top-level API
    "attach", "Trainer", "info",
    # Core classes
    "OtuxStore", "ImportanceWeights",
    "GradShield", "StabilityCore", "ModelStabilizer",
    "NipGraph",
    "WeightVault", "WeightSurgeon", "WeightGuard",
    "SafeMath", "safe_loss", "safe_divide",
    "AutoHeal",
    # Subpackages
    "memory", "stabilize", "monitor", "weights", "compute", "compat", "utils",
    "__version__",
]
