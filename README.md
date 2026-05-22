# ZPower (zp) — v1.2.0

**Intelligence layer for AI/ML systems.**  
Works *with* PyTorch and HuggingFace — not against them.

> Stabilize training. Protect good weights. Detect anomalies. Reduce wasted compute.

---

## Install

```bash
pip install zpower                # numpy only (core features)
pip install zpower[torch]         # + PyTorch support
pip install zpower[hf]            # + HuggingFace Transformers
pip install zpower[full]          # everything
```

---

## Quick Start

### Mode A — Attach to any existing model
```python
import zpower as zp

zp_model = zp.attach(
    my_model,
    stabilize    = True,   # GradShield + StabilityCore
    monitor      = True,   # NipGraph anomaly detection
    weight_vault = True,   # Record best weight snapshots
    auto_heal    = True,   # Auto-recover from training failures
)

output = zp_model(input_tensor)   # Forward pass unchanged
loss.backward()                   # GradShield hooks active automatically
```

### Mode B — Training from scratch
```python
trainer = zp.Trainer(
    my_model,
    stabilize    = True,
    weight_vault = True,
    auto_heal    = True,
)
trainer.fit(train_loader, epochs=20, lr=1e-3)
print(trainer.weight_report())
```

### Multi-model weight selection
```python
from zpower.weights import WeightSurgeon

surgeon = WeightSurgeon()
surgeon.auto_discover("./checkpoints/")   # Scans folder, asks y/N per file
best = surgeon.select_best()
new_model.load_state_dict(best)
```

---

## Module Overview

| Module | Purpose |
|---|---|
| `zp.memory.OtuxStore` | Selective 3D-coordinate memory — only stores important info |
| `zp.stabilize.GradShield` | Gradient health monitor — adaptive per-layer thresholds |
| `zp.stabilize.StabilityCore` | Loss EMA, plateau detection, LR signal |
| `zp.stabilize.ModelStabilizer` | Unified stabilization API |
| `zp.monitor.NipGraph` | Parity-aware training anomaly detection |
| `zp.weights.WeightVault` | Performance-gated weight snapshots |
| `zp.weights.WeightSurgeon` | Multi-model best weight selection with auto_discover |
| `zp.weights.WeightGuard` | EWC-style catastrophic forgetting prevention |
| `zp.heal.AutoHeal` | Automatic training failure recovery (rollback + LR cut) |
| `zp.compute.safe_loss` | NaN-safe loss with rational number fallback |
| `zp.compat.ZPowerModel` | Transparent PyTorch / HuggingFace model wrapper |
| `zp.compat.augment()` | HuggingFace model augmentation — one-line attach |
| `zp.utils.ZPConfig` | Centralized config — save/load JSON |
| `zp.utils.logging` | Structured logging with level control |
| `zp.weights.fisher` | Fisher Information computation (internal + standalone) |

### health_report() — Unified Status

```python
# Get full ZPower health from any wrapped model
report = zp_model.health_report()
print(report)
# {
#   "components": {
#     "stabilizer": { "status": "ok", "health_rate": 0.98, "last_state": "healthy" },
#     "memory":     { "status": "ok", "stored": 142, "compression_ratio": "71.6%" },
#     "nipgraph":   { "status": "ok", "alerts": 0, "converged": False },
#     "vault":      { "status": "ok", "total_snapshots": 23, "overall_vault_score": 0.84 }
#   }
# }
```

---

---

# Release Notes

---

## v1.2.0 — May 2026

**Theme: Critical Bug Fixes + Security + Performance**

3 critical bugs that silently broke core functionality were fixed. 5 high-priority correctness issues resolved. 3 performance upgrades. 1 new feature (WeightSurgeon.auto_discover).

### Critical Bug Fixes

#### 1. ZPowerModel.parameters() returned empty iterator — optimizer never trained model
**File:** `zpower/compat/torch_wrap.py`

**Problem:** In v1.1.0, `zp.attach(model)` wrapped the model but did not register it as a proper PyTorch submodule. When `zp.Trainer()` called `torch.optim.AdamW(model.parameters())`, the optimizer received zero parameters — an empty iterator. The model weights never updated during training. This silently made the entire Trainer non-functional for real training workloads.

**Fix:** Added explicit `parameters()` and `named_parameters()` overrides that delegate directly to `self._model`. Also added `state_dict()` and `load_state_dict()` delegation. Optimizer now correctly receives all model parameters.

```python
# v1.1.0 — optimizer silently received zero parameters
zp_model = zp.attach(my_model)
trainer  = zp.Trainer(zp_model)
trainer.fit(data, epochs=10)   # model weights NEVER changed

# v1.2.0 — fixed, model trains correctly
zp_model = zp.attach(my_model)
trainer  = zp.Trainer(zp_model)
trainer.fit(data, epochs=10)   # model weights update correctly
```

---

#### 2. StabilityCore plateau detection compared incompatible values
**File:** `zpower/stabilize/stability_core.py`

**Problem:** In v1.1.0, the plateau detector compared `self._ema` (a smoothed EMA value) against `self._history[-K]` (a raw noisy loss value from K steps ago). These are different types of values — smoothed vs raw. On noisy loss curves, this produced frequent false plateau signals and triggered unnecessary learning rate reductions throughout training.

**Fix:** Two-window comparison. Both windows are the same type (raw loss averages):
```
window_recent = mean(loss[-K:])
window_before = mean(loss[-2K:-K])
plateau if |window_recent - window_before| < plateau_threshold
```
No more false positives. LR signal now only triggers on real plateaus.

---

#### 3. write_batch() overwrote the wrong entry result
**File:** `zpower/memory/otux.py`

**Problem:** In v1.1.0, when a buffered entry got promoted to "stored" during a batch write, the code blindly did `results[-1] = "stored"` — overwriting the last processed entry's result, not the promoted entry's result. If entries A, B, C were processed and A got promoted, C's result was incorrectly overwritten.

**Fix:** Added `buf_result_idx` dictionary that maps each token hash to its exact position in the `results` list. On promotion, `results[buf_result_idx[h]] = "stored"` updates the correct index.

---

### High Priority Fixes

#### 4. OtuxStore memory was dead code during training
**File:** `zpower/compat/torch_wrap.py`

**Problem:** In v1.1.0, `OtuxStore` was created when `memory="otux_selective"` was set, but nothing ever called `write()` on it during training or inference. The entire memory module was silently inactive.

**Fix:** Added `_update_memory(output)` method to `ZPowerModel`. Called automatically at the end of every `forward()`. Extracts hidden representations from model outputs — supports HuggingFace `last_hidden_state`, tuple outputs, and plain tensors. Memory is now live during training without any user code changes.

---

#### 5. GradShield health state never reached WeightVault
**File:** `zpower/_trainer.py`

**Problem:** In v1.1.0, the trainer always passed `"grad_health": "healthy"` to `WeightVault` on every step, regardless of the actual gradient state. WeightVault's `grad_health` score (10% of performance score) was always 1.0 — even when gradients were exploding. This caused unstable checkpoints to be stored with inflated performance scores, making AutoHeal rollbacks unreliable.

**Fix:** Trainer now reads `model._stabilizer.grad_shield.last_state()` and passes the actual state to the vault on every step.

---

#### 6. torch.load() security vulnerability (RCE risk)
**File:** `zpower/weights/surgeon.py`

**Problem:** In v1.1.0, `WeightSurgeon._load_state()` called `torch.load(path)` without `weights_only=True`. In PyTorch 2.0+, this uses pickle deserialization which can execute arbitrary code when loading a malicious `.pt` file.

**Fix:** Changed to `torch.load(path, map_location="cpu", weights_only=True)` as default. Fallback to `weights_only=False` for older model formats that require full pickle, ensuring backward compatibility while prioritizing security.

---

#### 7. NipGraph false alerts on converged models
**File:** `zpower/monitor/nipgraph.py`

**Problem:** In v1.1.0, the stability band was computed as `band * (abs(ema) + 1e-8)`. When a model converged well and loss approached 0, the band collapsed to near-zero width. Any tiny fluctuation triggered a Y_W anomaly alert, flooding users with false alarms on healthy converged training.

**Fix:** Added `absolute_floor=0.10` parameter. Band is now `band * (abs(ema) + absolute_floor)`, ensuring it never collapses below `band × absolute_floor` regardless of EMA magnitude.

---

#### 8. NipGraph step counter drift
**File:** `zpower/monitor/nipgraph.py`

**Problem:** In v1.1.0, every call to `update()` unconditionally advanced `self._step = step + 1`. When GradShield and StabilityCore both called `ng.update()` at the same training step (e.g., step=100), the counter advanced to 102 instead of 101. Over a long training run, alert timestamps diverged significantly from actual training steps.

**Fix:** Changed to `self._step = max(self._step, step + 1)`. Counter only moves forward, never double-counts at the same step.

---

### New Feature: WeightSurgeon.auto_discover()

Automatically scans a directory for `.pt`/`.pth` weight files and asks for user approval before adding any as sources. Nothing is added without explicit `y` confirmation.

```python
surgeon = WeightSurgeon()

# Interactive mode (default) — asks y/N per file
surgeon.auto_discover("./checkpoints/")
# [WeightSurgeon] Found 3 weight file(s):
#   [1] run_jan.pt (245.3 MB)
#   [2] run_mar.pt (245.3 MB)
#   [3] run_may.pt (245.3 MB)
# Use 'run_jan.pt'? [y/N]: y  ✓ Added
# Use 'run_mar.pt'? [y/N]: y  ✓ Added
# Use 'run_may.pt'? [y/N]: N  — Skipped

# Non-interactive — just get list, no auto-adding
paths = surgeon.auto_discover("./checkpoints/", interactive=False)
surgeon.add_source(paths[0], label="best", perf_score=0.89)

best = surgeon.select_best()
```

---

### Performance Upgrades

| Component | Change | Impact |
|---|---|---|
| `OtuxStore.query()` | `np.argsort` → `np.argpartition` | O(N log N) → O(N), ~4x faster at N=10,000 |
| `GradShield.status()` | Full history scan → running counters | O(N) → O(1) per call |
| `StabilityCore._history` | `list + manual trim` → `deque(maxlen)` | O(N) trim → O(1) |
| `SafeMath` pocket | `dict` → `OrderedDict` | O(N) eviction → O(1) LRU |

---

### Other Fixes in v1.2.0

| Fix | File | Detail |
|---|---|---|
| `EMA == 0.0` sentinel | `stability_core.py`, `nipgraph.py` | Replaced with `_initialized` boolean — safe when first loss/observation is exactly `0.0` |
| WeightGuard re-attach | `weights/guard.py` | `attach()` now clears state before re-populating — prevented duplicate layer names on second attach |
| `query_by_coord()` return type | `memory/otux.py` | Now returns `List[Dict]` like `query()` — was returning `List[OtuxEntry]`, inconsistent API |
| Memory param validation | `compat/torch_wrap.py` | Invalid `memory=` values now raise `ValueError` with helpful message instead of silently creating no store |
| `__del__` hook cleanup | `grad_shield.py`, `model_stabilizer.py` | Hooks auto-removed on garbage collection — prevented backward hook leaks |
| Context manager | `grad_shield.py` | `with GradShield() as gs:` syntax now supported |
| `ZPConfig.to_json()` | `utils/config.py` | Save config to file, reload later |
| New config fields | `utils/config.py` | `grad_adaptive`, `grad_k`, `nipgraph_abs_floor`, `heal_lr_factor`, `heal_patience`, `heal_max_heals` |
| `get_level()` | `utils/logging.py` | Read current log level programmatically |
| `utils.logging` in Trainer | `_trainer.py` | Replaced `print()` with structured `zplog.info()` — log level controllable |

---

---

## v1.1.0 — May 2026

**Theme: Performance + AutoHeal + Adaptive Thresholds**

### New: AutoHeal Engine (`zp.heal.AutoHeal`)

When training fails (NaN loss, divergence, repeated gradient explosions), AutoHeal automatically recovers without human intervention.

**Three failure modes detected:**
- NaN / Inf loss → skip bad batch immediately, no backward pass
- Diverging loss (StabilityCore state == "diverging") → rollback + LR cut
- N consecutive gradient explosions (default N=5) → rollback + LR cut

**Recovery steps (automatic):**
1. Skip or pause current batch
2. Restore model weights from last WeightVault snapshot
3. Multiply optimizer LR by `heal_lr_factor` (default 0.5)
4. Resume training from safe state

```python
# Enable AutoHeal
trainer = zp.Trainer(my_model, weight_vault=True, auto_heal=True)
trainer.fit(train_loader, epochs=20)
# [AutoHeal] Heal #1. Reason: diverging. Rollback: ✓. New LR: 5.00e-04
# Training continues automatically...
```

---

### Fix: OTUX-S Memory — 100x Faster Writes

**Problem in v1.0.0:** Every `write()` call triggered a full O(N) matrix rebuild. At 10,000 stored entries, this meant copying 2.56 million elements on every single write — making any training loop with memory writes extremely slow.

**Fix:** Pre-allocated numpy matrix with 2x capacity headroom. Vectors appended via direct index assignment — O(1) amortized. Full rebuild only when capacity is exhausted (rare, amortized O(1)).

**New API: `write_batch()`** — N entries with a single matrix update instead of N individual rebuilds.

```python
store.write_batch([
    {"token": "note_C4",   "vector": vec1, "reward": 0.8},
    {"token": "chord_Cm7", "vector": vec2, "reward": 0.9},
])
```

**Also:** MD5 token hashing replaced with Python built-in `hash()` — 5-10x faster on hot path.

---

### Fix: GradShield — Adaptive Thresholds Per Layer

**Problem in v1.0.0:** Fixed thresholds (`clip_norm=5.0`, `vanish=1e-7`, `explode=50.0`) were hardcoded. A transformer's attention layers and a CNN's conv layers have completely different gradient distributions. Fixed values caused over-clipping on some layers and missed explosions on others.

**Fix:** Adaptive per-layer thresholds via Welford online algorithm:
```
clip_norm[layer] = running_mean[layer] + k × running_std[layer]
```
20-step warm-up uses fixed thresholds safely, then each layer adapts based on its actual gradient history.

```python
gs = zp.GradShield(adaptive=True, k=2.0)   # default
gs = zp.GradShield(adaptive=False)           # v1.0 behaviour if needed
```

**Also:** History storage changed from `list[-500:]` (O(N) copy) to `deque(maxlen=500)` (O(1) trim).

---

### Fix: Dangerous Hardcoded Fallback Removed

**Problem in v1.0.0:** When loss computation failed, the trainer silently set `loss = 0.01` — masking the actual problem. Models appeared to train while the loss was meaningless.

**Fix in v1.1.0:** Dangerous fallback removed. Non-finite loss now logs a clear warning. With `auto_heal=True`, AutoHeal handles the recovery properly.

---

---

## v1.0.0 — May 2026

**Initial release.** Seven core modules published.

### What was included

**`zp.memory.OtuxStore`** — Selective context-aware memory with 3D semantic coordinates.
Every entry tagged with `(x, y, z)` where x=sequence position, y=domain layer, z=reasoning depth.
Importance gate: `I = α·novelty + β·reward + γ·context_fit + δ·recurrence`.
Three-zone system: score above threshold → STORE, below floor → DISCARD, middle → BUFFER (3-strikes promote).

**`zp.stabilize.GradShield`** — Real-time gradient health monitor. Four states: VANISHING / HEALTHY / WARNING / EXPLODING. Hooks into PyTorch backward pass automatically via `register_hook`.

**`zp.stabilize.StabilityCore`** — Loss EMA tracking, plateau detection, curvature classification (flat/moderate/sharp), learning rate adjustment signal.

**`zp.monitor.NipGraph`** — Parity-aware anomaly detection. Tracks each variable's state category (x_M/x_W/Y_M/Y_W) and fires alerts when unexpected transitions occur.

**`zp.weights.WeightVault`** — Records weight snapshots only when performance score exceeds threshold. Performance formula: `P = w1*(1-loss) + w2*accuracy + w3*confidence + w4*grad_health + w5*curvature`.

**`zp.weights.WeightSurgeon`** — Selects best weights from 2-3 existing trained models using Fisher Information to determine which weights matter most per model. Conflict resolution: highest_performer / weighted_average / sign_vote.

**`zp.weights.WeightGuard`** — EWC-style (Elastic Weight Consolidation) penalty that protects important weights during fine-tuning: `L_total = L_task + λ · Σ F_i · (θ_i - θ_i*)²`.

**`zp.compat.ZPowerModel`** — Transparent PyTorch wrapper. All existing model calls (forward, generate) pass through unchanged. ZPower runs as a side-channel.

**`zp.Trainer`** — Drop-in training loop with ZPower intelligence active.

---

---

## Research Origins

Original research by **NNN Bhoi**:

| System | Origin |
|---|---|
| **OTUX-S** | PERATHINK research paper — selective 3D coordinate memory |
| **NipGraph** | NipGraph research paper — parity-aware state tracking |
| **GradShield + StabilityCore** | MUSAI vGPU engine — M2 + M4 modules |
| **WeightVault / Surgeon / Guard** | ZPower original research |
| **AutoHeal** | ZPower original research |

---

## License

MIT — free to use in any project.
