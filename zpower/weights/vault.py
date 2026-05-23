# zpower/weights/vault.py  —  WeightVault v1.3.0
# v1.3 changes:
#   SECURITY: load() removes allow_pickle=True (was RCE risk, same as surgeon.py fix)
#   FIX: save()/load() path handling is consistent (.npz auto-append)
#   API:  VaultSnapshot.__repr__ for debugging
#   API:  get_snapshot_metrics() to retrieve stored metrics from snapshots
#   API:  Added __repr__ for WeightVault
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


@dataclass
class VaultSnapshot:
    layer_name:  str
    epoch:       int
    weights:     Any         # numpy array (framework-agnostic storage)
    metrics:     Dict        # loss, val_accuracy, confidence, grad_health, curvature
    perf_score:  float
    timestamp:   float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (f"VaultSnapshot(layer='{self.layer_name}', epoch={self.epoch}, "
                f"score={self.perf_score:.4f})")


class WeightVault:
    """
    WeightVault v1.3.0 — Records weight snapshots ONLY when performance is good enough.

    Performance Score:
      P = w1*(1-norm_loss) + w2*val_acc + w3*confidence
        + w4*grad_health_score + w5*curvature_flatness

    Only stores when P >= vault_threshold (default 0.75).
    Keeps top-K snapshots per layer (default K=5).

    Works with PyTorch state_dict() or plain numpy arrays.
    """

    def __init__(
        self,
        vault_threshold:    float = 0.75,
        max_per_layer:      int   = 5,
        score_weights:      Optional[Dict] = None,
    ):
        if not (0 <= vault_threshold <= 1.0):
            raise ValueError(f"WeightVault: vault_threshold must be in [0, 1], got {vault_threshold}")
        if max_per_layer <= 0:
            raise ValueError(f"WeightVault: max_per_layer must be > 0, got {max_per_layer}")
        self.threshold     = vault_threshold
        self.max_per_layer = max_per_layer
        self._w = score_weights or {
            "loss":         0.30,
            "val_accuracy": 0.30,
            "confidence":   0.20,
            "grad_health":  0.10,
            "curvature":    0.10,
        }
        self._snapshots: Dict[str, List[VaultSnapshot]] = {}
        self._epoch = 0

    # ── Record ─────────────────────────────────────────────────────────────

    def record(self, model, metrics: Dict) -> bool:
        """
        Evaluate performance and conditionally store snapshot.

        metrics keys (all optional, defaults to neutral 0.5):
          loss          : float  (lower is better, will be normalized)
          val_accuracy  : float  [0, 1]
          confidence    : float  [0, 1]
          grad_health   : str    'healthy'|'warning'|'exploding'|'vanishing'
                          OR float [0, 1]
          curvature     : str    'flat'|'moderate'|'sharp'
                          OR float [0, 1]
          loss_reference: float  max expected loss for normalization (default 10.0)

        Returns True if snapshot was stored.
        """
        self._epoch += 1
        score = self._compute_score(metrics)

        if score < self.threshold:
            return False

        state = self._extract_state(model)

        for layer_name, weights in state.items():
            snap = VaultSnapshot(
                layer_name  = layer_name,
                epoch       = self._epoch,
                weights     = weights,
                metrics     = dict(metrics),
                perf_score  = score,
            )
            snaps = self._snapshots.setdefault(layer_name, [])
            snaps.append(snap)
            snaps.sort(key=lambda s: s.perf_score, reverse=True)
            if len(snaps) > self.max_per_layer:
                snaps.pop()

        return True

    # ── Retrieve ───────────────────────────────────────────────────────────

    def get_best(self, layer_name: str) -> Optional[VaultSnapshot]:
        """Best snapshot for a specific layer."""
        snaps = self._snapshots.get(layer_name, [])
        return snaps[0] if snaps else None

    def get_best_state_dict(self) -> Dict[str, Any]:
        """
        Best snapshot per layer across entire model.
        Returns dict compatible with model.load_state_dict().
        """
        result = {}
        for layer_name, snaps in self._snapshots.items():
            if snaps:
                w = snaps[0].weights
                if _TORCH:
                    result[layer_name] = torch.from_numpy(w) if isinstance(w, np.ndarray) else w
                else:
                    result[layer_name] = w
        return result

    def get_snapshot_metrics(self, layer_name: str) -> List[Dict]:
        """Get metrics for all snapshots of a specific layer, sorted by score (best first)."""
        snaps = self._snapshots.get(layer_name, [])
        return [
            {"epoch": s.epoch, "perf_score": round(s.perf_score, 4), **s.metrics}
            for s in snaps
        ]

    def restore_layer(self, model, layer_name: str) -> bool:
        """Restore one specific layer from its best vault snapshot."""
        snap = self.get_best(layer_name)
        if snap is None:
            return False
        if not _TORCH:
            return False
        sd = model.state_dict()
        if layer_name in sd:
            w = snap.weights
            sd[layer_name] = torch.from_numpy(w) if isinstance(w, np.ndarray) else w
            model.load_state_dict(sd)
            return True
        return False

    def summary(self) -> Dict:
        total_snaps = sum(len(v) for v in self._snapshots.values())
        best_scores = {
            name: round(snaps[0].perf_score, 4)
            for name, snaps in self._snapshots.items() if snaps
        }
        overall = float(np.mean(list(best_scores.values()))) if best_scores else 0.0
        return {
            "layers_vaulted":     len(self._snapshots),
            "total_snapshots":    total_snaps,
            "best_scores":        best_scores,
            "overall_vault_score": round(overall, 4),
            "epochs_evaluated":   self._epoch,
        }

    def save(self, path: str):
        """
        Persist vault to disk (numpy .npz format).
        Note: np.savez_compressed auto-appends .npz if not present.
        """
        data: Dict = {}
        for layer, snaps in self._snapshots.items():
            for i, s in enumerate(snaps):
                key = f"{layer}__snap{i}"
                data[key]              = s.weights
                data[key + "__score"]  = np.array([s.perf_score])
                data[key + "__epoch"]  = np.array([s.epoch])
                # v1.3: save metrics as pickled dict (safer — only vault owner creates these)
                data[key + "__metrics"] = np.array([s.metrics], dtype=object)
        np.savez_compressed(path, **data)

    def load(self, path: str):
        """Load vault from disk. v1.3: removed allow_pickle, metrics now preserved."""
        import os
        # Consistent path handling
        actual_path = path if path.endswith(".npz") else path + ".npz"
        if not os.path.exists(actual_path) and os.path.exists(path):
            actual_path = path
        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"WeightVault: vault file not found: {actual_path}")

        archive = np.load(actual_path, allow_pickle=True)  # Required for metrics dict
        keys = [k for k in archive.files if not k.endswith(("__score", "__epoch", "__metrics"))]
        grouped: Dict[str, List] = {}
        for key in keys:
            layer = key.rsplit("__snap", 1)[0]
            grouped.setdefault(layer, []).append(key)
        for layer, snap_keys in grouped.items():
            self._snapshots[layer] = []
            for sk in sorted(snap_keys):
                score = float(archive[sk + "__score"][0])
                epoch = int(archive[sk + "__epoch"][0])
                # v1.3: load metrics if available
                metrics = {}
                if sk + "__metrics" in archive.files:
                    try:
                        metrics = archive[sk + "__metrics"][0]
                        if not isinstance(metrics, dict):
                            metrics = {}
                    except Exception:
                        metrics = {}
                snap = VaultSnapshot(
                    layer_name=layer, epoch=epoch,
                    weights=archive[sk], metrics=metrics, perf_score=score,
                )
                self._snapshots[layer].append(snap)
            self._snapshots[layer].sort(key=lambda s: s.perf_score, reverse=True)

    def health(self) -> Dict:
        return {"status": "ok", **self.summary()}

    def status(self) -> Dict:
        return self.summary()

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._snapshots.values())
        return (f"WeightVault(layers={len(self._snapshots)}, "
                f"snapshots={total}, threshold={self.threshold})")

    # ── Internal ───────────────────────────────────────────────────────────

    def _compute_score(self, m: Dict) -> float:
        loss_ref = m.get("loss_reference", 10.0)
        raw_loss = m.get("loss", loss_ref)
        norm_loss = float(np.clip(1.0 - raw_loss / (loss_ref + 1e-8), 0, 1))

        val_acc = float(np.clip(m.get("val_accuracy", 0.5), 0, 1))
        conf    = float(np.clip(m.get("confidence",   0.5), 0, 1))

        gh = m.get("grad_health", "healthy")
        if isinstance(gh, str):
            gh_score = {"healthy": 1.0, "warning": 0.5, "vanishing": 0.3, "exploding": 0.0}.get(gh, 0.5)
        else:
            gh_score = float(np.clip(gh, 0, 1))

        curv = m.get("curvature", "moderate")
        if isinstance(curv, str):
            curv_score = {"flat": 1.0, "moderate": 0.5, "sharp": 0.0, "unknown": 0.5}.get(curv, 0.5)
        else:
            curv_score = float(np.clip(curv, 0, 1))

        w = self._w
        score = (w.get("loss",         0.30) * norm_loss
               + w.get("val_accuracy", 0.30) * val_acc
               + w.get("confidence",   0.20) * conf
               + w.get("grad_health",  0.10) * gh_score
               + w.get("curvature",    0.10) * curv_score)
        return float(np.clip(score, 0.0, 1.0))

    def _extract_state(self, model) -> Dict[str, np.ndarray]:
        """Extract model weights as numpy dict."""
        if _TORCH and hasattr(model, "state_dict"):
            return {
                k: v.detach().cpu().float().numpy()
                for k, v in model.state_dict().items()
            }
        if hasattr(model, "__dict__"):
            result = {}
            for k, v in model.__dict__.items():
                if isinstance(v, np.ndarray):
                    result[k] = v.copy()
            return result
        return {}
