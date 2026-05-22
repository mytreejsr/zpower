"""
ZPower Example 1 — Attach to any existing PyTorch model
"""
import torch
import torch.nn as nn
import zpower as zp

# Any existing model
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

# One line — attach ZPower intelligence
zp_model = zp.attach(
    model,
    stabilize    = True,   # GradShield + StabilityCore
    monitor      = True,   # NipGraph anomaly detection
    weight_vault = True,   # Record best weight snapshots
    auto_heal    = True,   # Auto-recover from failures
)

# Forward pass — completely unchanged
x      = torch.randn(32, 128)
output = zp_model(x)
loss   = output.mean()
loss.backward()   # GradShield hooks active here automatically

# Check ZPower status
print(zp_model.health_report())
