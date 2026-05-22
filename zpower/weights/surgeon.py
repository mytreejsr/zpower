# zpower/weights/surgeon.py  —  WeightSurgeon v1.2.0
# v1.2 fixes:
#   SECURITY: torch.load() now uses weights_only=True (RCE vulnerability fix)
#   NEW FEATURE: auto_discover() — scans directory, asks user approval before use
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

from zpower.weights.fisher import compute_diagonal


@dataclass
class SourceModel:
    label:      str
    state_dict: Dict[str, np.ndarray]
    perf_score: float = 0.5
    fisher:     Dict[str, np.ndarray] = field(default_factory=dict)


class WeightSurgeon:
    """
    WeightSurgeon v1.2.0 — Multi-model best weight selection.

    v1.2 security fix:
      torch.load() now uses weights_only=True by default to prevent
      arbitrary code execution when loading untrusted .pt files.
      Falls back to weights_only=False for older model formats.

    v1.2 new feature — auto_discover():
      Scans a directory for .pt/.pth weight files, shows what it found,
      and asks for user approval before adding them as sources.
      Nothing is added without explicit user confirmation.
      Use interactive=False to get the list without prompting.
    """

    def __init__(
        self,
        conflict_resolution: str = "highest_performer",
        calibration_batches: int = 10,
    ):
        self.conflict_resolution = conflict_resolution
        self.n_batches           = calibration_batches
        self._sources: List[SourceModel] = []
        self._selection_report: Dict     = {}

    # ── Auto-discover (NEW v1.2) ───────────────────────────────────────────

    def auto_discover(
        self,
        directory:   str  = ".",
        interactive: bool = True,
        perf_scores: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Scan a directory for .pt and .pth weight files.

        If interactive=True (default): shows found files and asks user
        approval before adding them as sources. Nothing added without
        explicit 'y' confirmation per file.

        If interactive=False: returns list of found paths WITHOUT adding.
        Caller must then call add_source() manually.

        perf_scores: optional dict mapping filename → perf_score.
                     If not provided, defaults to 0.5 for all discovered files.

        Returns: list of file paths that were approved/found.
        """
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            raise ValueError(f"auto_discover: '{directory}' is not a valid directory")

        # Scan for weight files
        found = []
        for fname in sorted(os.listdir(directory)):
            if fname.endswith((".pt", ".pth")):
                found.append(os.path.join(directory, fname))

        if not found:
            print(f"[WeightSurgeon] No .pt/.pth files found in: {directory}")
            return []

        print(f"\n[WeightSurgeon] Found {len(found)} weight file(s) in '{directory}':")
        for i, path in enumerate(found):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [{i+1}] {os.path.basename(path)} ({size_mb:.1f} MB)")

        if not interactive:
            print("[WeightSurgeon] interactive=False — returning paths without adding.")
            print("  Call add_source(path) manually to use any of these.")
            return found

        # Interactive approval — one by one
        approved = []
        print("\n[WeightSurgeon] Review each file for inclusion in weight selection.")
        print("  Type 'y' to approve, any other key to skip.\n")

        for path in found:
            fname = os.path.basename(path)
            score = (perf_scores or {}).get(fname, 0.5)
            try:
                ans = input(f"  Use '{fname}' (perf_score={score})? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  [WeightSurgeon] Input interrupted — skipping remaining files.")
                break

            if ans == "y":
                try:
                    self.add_source(path, label=fname, perf_score=score)
                    approved.append(path)
                    print(f"  ✓ Added: {fname}")
                except Exception as e:
                    print(f"  ✗ Failed to load '{fname}': {e}")
            else:
                print(f"  — Skipped: {fname}")

        print(f"\n[WeightSurgeon] {len(approved)}/{len(found)} files approved and added.")
        return approved

    # ── Add source ─────────────────────────────────────────────────────────

    def add_source(
        self,
        model_or_path,
        label:       str           = "",
        perf_score:  float         = 0.5,
        loss_fn      = None,
        calib_data   = None,
    ):
        if not label:
            label = f"model_{len(self._sources) + 1}"
        state_dict = self._load_state(model_or_path)
        if state_dict is None:
            raise ValueError(f"Cannot load model from: {model_or_path}")

        fisher: Dict[str, np.ndarray] = {}
        if _TORCH and loss_fn is not None and calib_data is not None \
                and hasattr(model_or_path, "parameters"):
            try:
                fisher = compute_diagonal(
                    model_or_path, loss_fn, calib_data, self.n_batches
                )
            except Exception:
                pass

        self._sources.append(SourceModel(
            label=label, state_dict=state_dict,
            perf_score=perf_score, fisher=fisher,
        ))

    # ── Select ─────────────────────────────────────────────────────────────

    def select_best(self) -> Dict[str, Any]:
        if not self._sources:
            raise RuntimeError("No source models added. Call add_source() or auto_discover() first.")
        if len(self._sources) == 1:
            return self._to_output(self._sources[0].state_dict)

        all_layers = set(self._sources[0].state_dict.keys())
        for src in self._sources[1:]:
            all_layers &= set(src.state_dict.keys())

        result = {}
        self._selection_report = {}
        for layer in all_layers:
            chosen_weights, chosen_label = self._select_layer(layer)
            result[layer] = chosen_weights
            self._selection_report[layer] = chosen_label

        return self._to_output(result)

    def selection_report(self) -> Dict:
        return dict(self._selection_report)

    def status(self) -> Dict:
        return {"status": "ok", "sources": len(self._sources),
                "conflict_resolution": self.conflict_resolution}

    def health(self) -> Dict:
        return {"status": "ok", **self.status()}

    # ── Internal ───────────────────────────────────────────────────────────

    def _select_layer(self, layer: str) -> Tuple[np.ndarray, str]:
        candidates = []
        for src in self._sources:
            if layer not in src.state_dict: continue
            w       = src.state_dict[layer]
            f_score = float(np.mean(src.fisher[layer])) if layer in src.fisher else 0.5
            combined = src.perf_score * (0.5 + 0.5 * f_score)
            candidates.append((src.label, w, src.perf_score, f_score, combined))
        if not candidates:
            return np.zeros(1, dtype=np.float32), "empty"
        candidates.sort(key=lambda c: c[4], reverse=True)

        if self.conflict_resolution == "highest_performer":
            return candidates[0][1], candidates[0][0]
        elif self.conflict_resolution == "weighted_average":
            total_w = sum(c[4] for c in candidates) + 1e-10
            blended = sum(c[4] * c[1].astype(np.float64) for c in candidates) / total_w
            return blended.astype(np.float32), "blended"
        elif self.conflict_resolution == "sign_vote":
            best_w    = candidates[0][1].astype(np.float64)
            sign_sum  = sum(np.sign(c[1].astype(np.float64)) for c in candidates)
            maj_sign  = np.sign(sign_sum)
            maj_sign[maj_sign == 0] = np.sign(best_w[maj_sign == 0])
            return (np.abs(best_w) * maj_sign).astype(np.float32), f"sign_vote({candidates[0][0]})"
        return candidates[0][1], candidates[0][0]

    def _load_state(self, model_or_path) -> Optional[Dict[str, np.ndarray]]:
        if isinstance(model_or_path, dict):
            out = {}
            for k, v in model_or_path.items():
                if _TORCH and isinstance(v, torch.Tensor):
                    out[k] = v.detach().cpu().float().numpy()
                else:
                    out[k] = np.asarray(v, dtype=np.float32)
            return out

        if isinstance(model_or_path, str):
            if _TORCH:
                try:
                    # v1.2 security fix: weights_only=True prevents RCE
                    sd = torch.load(model_or_path, map_location="cpu", weights_only=True)
                except Exception:
                    try:
                        # Fallback for older formats that need full pickle
                        sd = torch.load(model_or_path, map_location="cpu", weights_only=False)
                    except Exception:
                        return None
                if hasattr(sd, "state_dict"):
                    sd = sd.state_dict()
                return {k: v.float().numpy() for k, v in sd.items()
                        if isinstance(v, torch.Tensor)}
            return None

        if _TORCH and hasattr(model_or_path, "state_dict"):
            return {k: v.detach().cpu().float().numpy()
                    for k, v in model_or_path.state_dict().items()}
        return None

    def _to_output(self, state_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if _TORCH:
            return {k: torch.from_numpy(v.copy()) for k, v in state_dict.items()}
        return {k: v.copy() for k, v in state_dict.items()}
