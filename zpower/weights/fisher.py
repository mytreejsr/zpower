# zpower/weights/fisher.py  —  Diagonal Fisher Information (internal)
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


def compute_diagonal(
    model,
    loss_fn:          Callable,
    calibration_data,
    n_batches:        int = 10,
    device:           Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute diagonal Fisher Information for all named parameters.

    F_i ~ E[(dL/d_theta_i)^2] — estimated as mean squared gradient
    over calibration_data batches.

    Returns: { param_name: np.ndarray of same shape as param }

    High F_i -> this weight is critically important -> protect it.
    Low  F_i -> this weight is replaceable.
    """
    if not _TORCH:
        raise ImportError("Fisher computation requires torch")

    model.eval()
    fisher: Dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)

    count = 0
    for i, batch in enumerate(calibration_data):
        if i >= n_batches:
            break
        try:
            model.zero_grad()

            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                inputs  = batch
                targets = None

            if device:
                if hasattr(inputs, "to"):
                    inputs = inputs.to(device)
                if targets is not None and hasattr(targets, "to"):
                    targets = targets.to(device)

            outputs = model(inputs)

            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                log_prob = torch.nn.functional.log_softmax(outputs, dim=-1)
                loss     = -log_prob.mean()

            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            count += 1

        except Exception:
            continue

    if count > 0:
        for name in fisher:
            fisher[name] /= count

    return {name: f.cpu().numpy() for name, f in fisher.items()}


def fisher_importance_score(fisher_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Aggregate Fisher per parameter -> single importance score per layer.
    Used by WeightSurgeon for layer-level selection decisions.
    """
    scores = {}
    for name, f in fisher_dict.items():
        scores[name] = float(np.mean(f))
    return scores
