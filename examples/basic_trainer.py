"""
ZPower Example 2 — Training from scratch with ZPower Trainer
"""
import torch
import torch.nn as nn
import zpower as zp

model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))

# Fake dataset
dataset = [(torch.randn(16, 64), torch.randint(0, 10, (16,))) for _ in range(20)]

trainer = zp.Trainer(
    model,
    stabilize    = True,
    weight_vault = True,
    auto_heal    = True,
)

history = trainer.fit(dataset, epochs=5, lr=1e-3)
print(trainer.weight_report())
trainer.render_monitor()
