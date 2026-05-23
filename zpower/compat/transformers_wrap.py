# zpower/compat/transformers_wrap.py  —  HuggingFace model augmentation
from __future__ import annotations
from typing import Any, Dict, Optional


def augment(
    model,
    memory:       str  = "otux_selective",
    stabilize:    bool = True,
    monitor:      bool = True,
    weight_vault: bool = False,
    weight_guard: bool = False,
    auto_heal:    bool = False,
):
    """
    Augment any HuggingFace PreTrainedModel with ZPower.

    Returns a ZPowerModel that passes through all HuggingFace-specific
    methods (generate, from_pretrained, save_pretrained, etc.).

    Example:
        from transformers import AutoModelForCausalLM
        import zpower as zp

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        zp_model = zp.compat.augment(model, stabilize=True, weight_vault=True)
        outputs = zp_model.generate(input_ids, max_new_tokens=50)
    """
    from zpower.compat.torch_wrap import ZPowerModel
    return ZPowerModel(
        model,
        memory       = memory,
        stabilize    = stabilize,
        monitor      = monitor,
        weight_vault = weight_vault,
        weight_guard = weight_guard,
        auto_heal    = auto_heal,
    )
