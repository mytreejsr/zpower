# zpower/compat/torch_wrap.py  —  ZPowerModel v1.3.0
# v1.3 changes:
#   FIX:  _step incremented in forward() (was only in zp_on_step_end,
#         causing all memory entries to be "step_0" without Trainer)
#   API:  pprint_status() for formatted console output
#   API:  overhead tracking — measures zpower overhead per forward pass
#   API:  Added __repr__
from __future__ import annotations
import time
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
from zpower.utils import logging as zplog

_VALID_MEMORY_MODES = {"otux_selective", "otux_full", "none"}


class ZPowerModel(nn.Module if _TORCH else object):
    """
    ZPowerModel v1.3.0 — wraps any PyTorch nn.Module with ZPower intelligence.

    v1.2: parameter delegation, memory update in forward, input validation.
    v1.3: _step now incremented in forward() for correct memory entries,
          pprint_status() for formatted output, overhead tracking, __repr__.
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

        if memory not in _VALID_MEMORY_MODES:
            raise ValueError(
                f"ZPowerModel: invalid memory='{memory}'. "
                f"Valid options: {sorted(_VALID_MEMORY_MODES)}"
            )

        self.add_module("_inner", model)
        self._model      = model
        self._step       = 0
        self._auto_heal  = auto_heal

        # v1.3: overhead tracking
        self._zp_overhead_ms: float = 0.0
        self._zp_forward_count: int = 0

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

    # ── Parameter delegation ───────────────────────────────────────────────

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
        t0 = time.perf_counter()

        output = self._model(*args, **kwargs)

        # v1.3: increment _step in forward() so memory entries have correct step numbers
        self._step += 1

        if self._memory is not None:
            self._update_memory(output)

        # v1.3: track overhead
        elapsed = (time.perf_counter() - t0) * 1000
        self._zp_overhead_ms += max(0, elapsed)
        self._zp_forward_count += 1

        return output

    # ── Memory update ──────────────────────────────────────────────────────

    def _update_memory(self, output):
        """
        Extract hidden representation from model output and write to OtuxStore.
        Supports: HuggingFace (last_hidden_state), tuple outputs, plain tensors.
        """
        try:
            if hasattr(output, "last_hidden_state"):
                hidden = output.last_hidden_state
            elif isinstance(output, (tuple, list)):
                hidden = output[0]
            else:
                hidden = output

            if not isinstance(hidden, torch.Tensor):
                return

            if hidden.dim() >= 2:
                rep = hidden.detach().mean(dim=list(range(hidden.dim() - 1)))
            else:
                rep = hidden.detach()

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
        # Note: _step is now incremented in forward(), not here
        if self._vault and metrics:
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
                pass

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
        report = export_health(components)
        report["overhead_ms"] = round(self._zp_overhead_ms, 3)
        report["forward_count"] = self._zp_forward_count
        return report

    def zp_status(self) -> Dict:
        status: Dict = {"step": self._step}
        if self._stabilizer: status["stabilizer"] = self._stabilizer.status()
        if self._memory:     status["memory"]     = self._memory.filter_stats()
        if self._nipgraph:   status["nipgraph"]   = self._nipgraph.status()
        if self._vault:      status["vault"]      = self._vault.summary()
        if self._guard:      status["guard"]      = self._guard.status()
        status["overhead_ms"] = round(self._zp_overhead_ms, 3)
        status["forward_count"] = self._zp_forward_count
        return status

    def pprint_status(self):
        """Print formatted ZPower status to console."""
        status = self.zp_status()
        print(f"\n{'='*58}")
        print(f"  ZPower Status — {type(self._model).__name__}")
        print(f"{'='*58}")
        print(f"  Step:              {status.get('step', 0)}")
        print(f"  Forward calls:     {status.get('forward_count', 0)}")
        overhead = status.get('overhead_ms', 0)
        count = max(status.get('forward_count', 1), 1)
        print(f"  ZPower overhead:   {overhead:.2f}ms total "
              f"({overhead/count:.3f}ms/call)")

        if "stabilizer" in status:
            gs = status["stabilizer"]["grad_shield"]
            sc = status["stabilizer"]["stability_core"]
            print(f"\n  GradShield:")
            print(f"    health_rate:    {gs.get('health_rate', 'N/A')}")
            print(f"    last_state:     {gs.get('last_state', 'N/A')}")
            print(f"    layers_tracked: {gs.get('layers_tracked', 0)}")
            print(f"  StabilityCore:")
            print(f"    ema:            {sc.get('ema', 'N/A')}")
            print(f"    curvature:      {sc.get('curvature', 'N/A')}")
            print(f"    lr_signal:      {sc.get('lr_signal', 'N/A')}")

        if "memory" in status:
            m = status["memory"]
            print(f"\n  OtuxStore Memory:")
            print(f"    stored:         {m.get('currently_stored', 0)}")
            print(f"    compression:    {m.get('compression_ratio', 'N/A')}")

        if "nipgraph" in status:
            n = status["nipgraph"]
            print(f"\n  NipGraph:")
            print(f"    converged:      {n.get('converged', False)}")
            print(f"    alerts:         {n.get('alerts', 0)}")

        if "vault" in status:
            v = status["vault"]
            print(f"\n  WeightVault:")
            print(f"    layers_vaulted: {v.get('layers_vaulted', 0)}")
            print(f"    total_snaps:    {v.get('total_snapshots', 0)}")
            print(f"    vault_score:    {v.get('overall_vault_score', 'N/A')}")

        if "guard" in status:
            g = status["guard"]
            print(f"\n  WeightGuard:")
            print(f"    protected:      {g.get('protected_layers', 0)}")
            print(f"    free:           {g.get('free_layers', 0)}")

        print(f"{'='*58}\n")

    def overhead_ms(self) -> float:
        """Total zpower overhead in milliseconds across all forward calls."""
        return self._zp_overhead_ms

    def overhead_per_call_ms(self) -> float:
        """Average zpower overhead per forward call in milliseconds."""
        if self._zp_forward_count == 0:
            return 0.0
        return self._zp_overhead_ms / self._zp_forward_count

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
        """Detach all ZPower components and return the original model."""
        if self._stabilizer:
            self._stabilizer.detach()
        return self._model

    def __repr__(self) -> str:
        return (f"ZPowerModel({type(self._model).__name__}, "
                f"step={self._step}, "
                f"stabilizer={'on' if self._stabilizer else 'off'}, "
                f"memory={'on' if self._memory else 'off'}, "
                f"vault={'on' if self._vault else 'off'})")

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
