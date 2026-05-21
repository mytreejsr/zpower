# zpower/compat/torch_wrap.py  —  ZPowerModel v1.2.0
# v1.2 CRITICAL fixes:
#   parameters() / named_parameters() delegation to inner model
#   state_dict() / load_state_dict() delegation
#   OtuxStore _update_memory() called from forward() — memory no longer dead code
#   memory parameter validated against allowed set
#   health_report() unified aggregation
#   __del__ for hook cleanup
#   auto_attach_guard() for one-call Fisher + guard setup
from __future__ import annotations
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

from zpower.memory.otux             import OtuxStore
from zpower.stabilize.model_stabilizer import ModelStabilizer
from zpower.monitor.nipgraph        import NipGraph
from zpower.weights.vault           import WeightVault
from zpower.weights.guard           import WeightGuard
from zpower.utils                   import export_health

_VALID_MEMORY_MODES = {"otux_selective", "otux_full", "none"}


class ZPowerModel(nn.Module if _TORCH else object):
    """
    ZPowerModel v1.2.0 — wraps any PyTorch nn.Module with ZPower intelligence.

    v1.2 critical fixes:
      parameters() / named_parameters() delegate to inner model — optimizer now
        receives actual parameters (was returning empty iterator in v1.1).
      state_dict() / load_state_dict() delegate to inner model.
      OtuxStore now populated during forward() via _update_memory().
      memory parameter validated — typos raise clear ValueError.
      health_report() aggregates all active components.
      __del__ removes hooks to prevent leaks.
    """

    def __init__(
        self,
        model,
        memory:       str  = "otux_selective",
        stabilize:    bool = True,
        monitor:      bool = True,
        weight_vault: bool = False,
        weight_guard: bool = False,
        auto_heal:    bool = False,
        otux_dim:     int  = 256,
    ):
        if not _TORCH:
            raise ImportError("ZPowerModel requires torch")

        super().__init__()

        # v1.2: validate memory parameter — no silent typo failures
        if memory not in _VALID_MEMORY_MODES:
            raise ValueError(
                f"ZPowerModel: invalid memory='{memory}'. "
                f"Valid options: {sorted(_VALID_MEMORY_MODES)}"
            )

        # Store inner model as registered submodule so nn.Module tracks it
        # v1.2: use add_module() for proper parameter registration
        self.add_module("_inner", model)
        self._model      = model
        self._step       = 0
        self._auto_heal  = auto_heal

        # NipGraph
        self._nipgraph: Optional[NipGraph] = None
        if monitor:
            self._nipgraph = NipGraph(
                variables=["loss", "loss_ema", "grad_norm"],
                band_width=0.10, absolute_floor=0.10,
            )

        # Stabilizer
        self._stabilizer: Optional[ModelStabilizer] = None
        if stabilize:
            self._stabilizer = ModelStabilizer(nipgraph=self._nipgraph)
            self._stabilizer.attach(model)

        # Memory
        self._memory: Optional[OtuxStore] = None
        if memory != "none":
            mode = "selective" if memory == "otux_selective" else "full"
            self._memory = OtuxStore(dim=otux_dim, mode=mode)

        # Vault
        self._vault: Optional[WeightVault] = None
        if weight_vault:
            self._vault = WeightVault()

        # Guard
        self._guard: Optional[WeightGuard] = None
        if weight_guard and self._vault is not None:
            self._guard = WeightGuard(vault=self._vault)

    # ── CRITICAL v1.2: parameter delegation ───────────────────────────────

    def parameters(self, recurse: bool = True):
        """Delegate to inner model — optimizer receives actual parameters."""
        return self._model.parameters(recurse)

    def named_parameters(self, *args, **kwargs):
        """Delegate to inner model."""
        return self._model.named_parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Delegate to inner model."""
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Delegate to inner model."""
        return self._model.load_state_dict(state_dict, strict=strict)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, *args, **kwargs):
        """Pass through to inner model. Updates OtuxStore memory automatically."""
        output = self._model(*args, **kwargs)
        # v1.2: OTUX-S memory updated on every forward — no longer dead code
        if self._memory is not None:
            self._update_memory(output)
        return output

    # ── Memory update (v1.2 new) ───────────────────────────────────────────

    def _update_memory(self, output):
        """
        Extract hidden representation from model output and write to OtuxStore.
        Supports: HuggingFace (last_hidden_state), tuple outputs, plain tensors.
        """
        try:
            if hasattr(output, "last_hidden_state"):
                # HuggingFace model output
                hidden = output.last_hidden_state
            elif isinstance(output, (tuple, list)):
                hidden = output[0]
            else:
                hidden = output

            if not isinstance(hidden, torch.Tensor):
                return

            # Take mean pooled representation
            if hidden.dim() >= 2:
                rep = hidden.detach().mean(dim=list(range(hidden.dim() - 1)))
            else:
                rep = hidden.detach()

            # Resize to OTUX dim if needed
            dim = self._memory.dim
            if rep.numel() != dim:
                rep_flat = rep.flatten()
                if rep_flat.numel() >= dim:
                    rep = rep_flat[:dim]
                else:
                    padded = torch.zeros(dim)
                    padded[:rep_flat.numel()] = rep_flat
                    rep = padded

            self._memory.write(
                token  = f"step_{self._step}",
                vector = rep.cpu().float().numpy(),
                x      = float(self._step),
                reward = 0.5,
            )
        except Exception:
            pass   # Memory update is best-effort — never crash forward()

    # ── ZPower hooks (called by Trainer) ──────────────────────────────────

    def zp_on_loss(self, loss_value: float, optimizer=None) -> Dict:
        info = {}
        if self._stabilizer:
            info = self._stabilizer.on_loss(loss_value)
            if optimizer and info.get("lr_signal", 1.0) < 0.99:
                self._stabilizer.apply_lr_signal(optimizer)
        return info

    def zp_on_step_end(self, metrics: Optional[Dict] = None):
        self._step += 1
        if self._vault and metrics:
            # v1.2: fill grad_health from GradShield if not provided
            if "grad_health" not in metrics and self._stabilizer:
                gs_state = self._stabilizer.grad_shield.last_state()
                if gs_state:
                    metrics["grad_health"] = gs_state
            self._vault.record(self._model, metrics)

    def ewc_penalty(self):
        if self._guard:
            return self._guard.ewc_penalty()
        if _TORCH:
            return torch.tensor(0.0)
        return 0.0

    def auto_attach_guard(self, calib_data=None, loss_fn=None):
        """
        One-call guard setup: auto-computes Fisher if calib_data provided,
        then attaches WeightGuard. Falls back to uniform Fisher if fails.
        """
        if self._vault is None:
            raise RuntimeError("auto_attach_guard requires weight_vault=True")
        if self._guard is None:
            self._guard = WeightGuard(vault=self._vault)

        fisher_dict = None
        if calib_data is not None and loss_fn is not None:
            try:
                from zpower.weights.fisher import compute_diagonal
                fisher_dict = compute_diagonal(
                    self._model, loss_fn, calib_data
                )
            except Exception:
                pass   # uniform Fisher fallback

        self._guard.attach(self._model, fisher_dict)
        return self._guard

    # ── Health & status ────────────────────────────────────────────────────

    def health_report(self) -> Dict:
        """Unified health report from all active ZPower components."""
        components = {}
        if self._stabilizer: components["stabilizer"] = self._stabilizer
        if self._memory:     components["memory"]     = self._memory
        if self._nipgraph:   components["nipgraph"]   = self._nipgraph
        if self._vault:      components["vault"]      = self._vault
        if self._guard:      components["guard"]      = self._guard
        return export_health(components)

    def zp_status(self) -> Dict:
        status: Dict = {"step": self._step}
        if self._stabilizer: status["stabilizer"] = self._stabilizer.status()
        if self._memory:     status["memory"]     = self._memory.filter_stats()
        if self._nipgraph:   status["nipgraph"]   = self._nipgraph.status()
        if self._vault:      status["vault"]      = self._vault.summary()
        if self._guard:      status["guard"]      = self._guard.status()
        return status

    @property
    def original_model(self):   return self._model
    @property
    def stabilizer(self):       return self._stabilizer
    @property
    def memory(self):           return self._memory
    @property
    def monitor(self):          return self._nipgraph
    @property
    def vault(self):            return self._vault
    @property
    def guard(self):            return self._guard

    def detach_zpower(self):
        if self._stabilizer:
            self._stabilizer.detach()
        return self._model

    def __del__(self):
        try:
            if self._stabilizer:
                self._stabilizer.detach()
        except Exception:
            pass

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
