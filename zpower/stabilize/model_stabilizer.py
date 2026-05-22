# zpower/stabilize/model_stabilizer.py  —  ModelStabilizer v1.2.0
# v1.2: health() API, __del__ cleanup
from __future__ import annotations
from typing import Any, Dict, Optional

from zpower.stabilize.grad_shield    import GradShield
from zpower.stabilize.stability_core import StabilityCore


class ModelStabilizer:
    """ModelStabilizer v1.2.0 — Unified GradShield + StabilityCore."""

    def __init__(self, grad_clip: float = 5.0, ema_beta: float = 0.95, nipgraph=None):
        self._nipgraph    = nipgraph
        self.grad_shield  = GradShield(clip_norm=grad_clip, nipgraph=nipgraph)
        self.stability_core = StabilityCore(ema_beta=ema_beta, nipgraph=nipgraph)
        self._model: Optional[Any] = None

    def attach(self, model) -> "ModelStabilizer":
        self._model = model
        self.grad_shield.attach_to_model(model)
        return self

    def on_loss(self, loss_value: float) -> Dict:
        return self.stability_core.update(loss_value)

    def on_backward(self, gradients=None):
        if gradients is not None:
            return self.grad_shield.shield(gradients)

    def detach(self):
        self.grad_shield.detach()
        self._model = None

    def apply_lr_signal(self, optimizer) -> float:
        signal = self.stability_core.get_lr_signal()
        if signal < 0.99:
            try:
                for group in optimizer.param_groups:
                    group["lr"] *= signal
            except AttributeError:
                pass
        return signal

    def status(self) -> Dict:
        return {
            "grad_shield":    self.grad_shield.status(),
            "stability_core": self.stability_core.status(),
        }

    def health(self) -> Dict:
        return {"status": "ok", **self.status()}

    def __del__(self):
        try: self.grad_shield.detach()
        except Exception: pass
