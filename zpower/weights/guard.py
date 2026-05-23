# zpower/weights/guard.py  —  WeightGuard v1.3.0
# v1.3: Added __repr__, input validation
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

from zpower.weights.vault import WeightVault


class WeightGuard:
    """
    WeightGuard v1.3.0 — EWC-style catastrophic forgetting prevention.

    v1.2 fix: attach() clears state before re-populating to prevent duplication.
    v1.3: Added __repr__, validation for lambda and adapt_rate.
    """

    def __init__(
        self,
        vault:               WeightVault,
        protection_strength: float = 0.8,
        adapt_rate:          float = 0.1,
    ):
        if not _TORCH:
            raise ImportError("WeightGuard requires torch")
        if protection_strength < 0:
            raise ValueError(f"WeightGuard: protection_strength must be >= 0, got {protection_strength}")
        if not (0 <= adapt_rate <= 1):
            raise ValueError(f"WeightGuard: adapt_rate must be in [0, 1], got {adapt_rate}")
        self._vault     = vault
        self.lambda_    = protection_strength
        self.adapt      = adapt_rate
        self._model: Optional[Any]              = None
        self._optimal:  Dict[str, torch.Tensor] = {}
        self._fisher:   Dict[str, torch.Tensor] = {}
        self._protected: List[str]              = []
        self._free:      List[str]              = []

    def attach(self, model, fisher_dict: Optional[Dict] = None) -> "WeightGuard":
        """
        Attach guard to model.
        Clears all state before re-populating to prevent duplication.
        """
        self._optimal.clear()
        self._fisher.clear()
        self._protected.clear()
        self._free.clear()

        self._model   = model
        best_sd       = self._vault.get_best_state_dict()
        if not best_sd:
            return self

        device = next(model.parameters()).device

        for name, param in model.named_parameters():
            if name in best_sd:
                opt = best_sd[name]
                if isinstance(opt, np.ndarray):
                    opt = torch.from_numpy(opt)
                self._optimal[name] = opt.to(device).float()

                if fisher_dict and name in fisher_dict:
                    f = fisher_dict[name]
                    if isinstance(f, np.ndarray):
                        f = torch.from_numpy(f)
                    self._fisher[name] = f.to(device).float()
                else:
                    self._fisher[name] = torch.ones_like(param.data)

                self._protected.append(name)
            else:
                self._free.append(name)
        return self

    def ewc_penalty(self) -> "torch.Tensor":
        if not _TORCH or self._model is None or not self._optimal:
            return torch.tensor(0.0, requires_grad=False)
        device  = next(self._model.parameters()).device
        penalty = torch.zeros(1, device=device)
        for name, param in self._model.named_parameters():
            if name in self._optimal:
                diff    = param - self._optimal[name]
                fisher  = self._fisher.get(name, torch.ones_like(param))
                strength = self.lambda_ * (1.0 - self.adapt)
                penalty  = penalty + (fisher * diff.pow(2)).sum() * strength
        return penalty

    def protected_layers(self) -> List[str]: return list(self._protected)
    def free_layers(self)      -> List[str]: return list(self._free)

    def update_vault_reference(self, vault: WeightVault):
        self._vault = vault
        if self._model is not None:
            self.attach(self._model)

    def status(self) -> Dict:
        return {
            "status":            "ok",
            "protection_lambda": self.lambda_,
            "adapt_rate":        self.adapt,
            "protected_layers":  len(self._protected),
            "free_layers":       len(self._free),
            "model_attached":    self._model is not None,
        }

    def health(self) -> Dict:
        return {"status": "ok", **self.status()}

    def __repr__(self) -> str:
        return (f"WeightGuard(lambda={self.lambda_}, adapt={self.adapt}, "
                f"protected={len(self._protected)}, free={len(self._free)})")
