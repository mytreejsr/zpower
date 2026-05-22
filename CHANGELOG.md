# ZPower Changelog

All notable changes to ZPower are documented here.

---

## [1.2.0] — May 2026

### Critical Bug Fixes
- **CRITICAL** `ZPowerModel.parameters()` returned empty iterator — optimizer never updated model weights. Fixed with explicit delegation to inner model.
- **CRITICAL** `StabilityCore` plateau detection compared EMA (smoothed) vs raw loss (noisy) — false positives caused wrong LR reductions. Fixed with two-window raw average comparison.
- **CRITICAL** `write_batch()` overwrote wrong result index on buffer promotion. Fixed with `buf_result_idx` tracking.

### Security
- **SECURITY** `WeightSurgeon` used `torch.load()` without `weights_only=True` — RCE vulnerability when loading untrusted `.pt` files. Fixed: `weights_only=True` default, fallback for legacy formats.

### High Priority Fixes
- `OtuxStore` memory was never written during training — dead code. Fixed: `_update_memory()` called from `forward()` automatically.
- `GradShield` state was never passed to `WeightVault` — always "healthy". Fixed: actual state extracted and passed on each step.
- `NipGraph` false alerts on converged models (near-zero EMA). Fixed: `absolute_floor=0.10` prevents band collapse.
- `NipGraph` step counter drift when multiple variables updated at same step. Fixed: `max()` instead of increment.

### New Features
- `WeightSurgeon.auto_discover()` — scans directory for `.pt`/`.pth` files, asks user approval per file. Nothing added without `y` confirmation.
- `ZPowerModel.health_report()` — unified health aggregation from all active components.
- `ZPowerModel.auto_attach_guard()` — one-call Fisher computation + WeightGuard setup.
- `ZPConfig.to_json()` — save config to file and reload.
- New config fields: `grad_adaptive`, `grad_k`, `nipgraph_abs_floor`, `heal_lr_factor`, `heal_patience`, `heal_max_heals`.

### Performance
- `OtuxStore.query()`: O(N log N) → O(N) via `np.argpartition` (~4x at N=10,000)
- `GradShield.status()` health_rate: O(N) scan → O(1) running counters
- `StabilityCore._history`: `list + trim` → `deque(maxlen)` O(1) append
- `SafeMath` pocket: `dict` → `OrderedDict` for O(1) LRU eviction

### Other Fixes
- `EMA == 0.0` sentinel → `_initialized` boolean in `StabilityCore` and `NipGraph`
- `WeightGuard.attach()` clears state before re-populating (prevented duplicate layers)
- `query_by_coord()` returns consistent `List[Dict]` format (was `List[OtuxEntry]`)
- Memory parameter validated — typos raise `ValueError` with helpful message
- `GradShield` + `ModelStabilizer` implement `__del__` and context manager
- `utils.logging` used in `_trainer.py` instead of bare `print()`
- `pyproject.toml` GitHub URL corrected to `mytreejsr/zpower`
- PyPI classifiers and keywords added for discoverability

---

## [1.1.0] — May 2026

### New Features
- **AutoHeal Engine** (`zp.heal.AutoHeal`) — automatic training failure recovery
  - NaN loss → skip bad batch
  - Diverging loss → rollback weights from WeightVault + reduce LR
  - N consecutive gradient explosions → rollback + reduce LR
  - Configurable: `heal_lr_factor`, `explode_patience`, `max_heals`

### Performance
- `OtuxStore.write()`: O(N) full matrix rebuild → O(1) amortized via pre-allocated matrix with 2x capacity headroom
- `OtuxStore.write_batch()`: new API — N entries with single matrix update
- Token hashing: MD5 → Python built-in `hash()` (5-10x faster)

### Fixes
- `GradShield` thresholds: fixed hardcoded values → adaptive per-layer (Welford online algorithm)
  - `clip_norm[layer] = mean[layer] + k × std[layer]`
  - 20-step warm-up uses fixed thresholds safely
- `GradShield` history: `list[-500:]` (O(N)) → `deque(maxlen=500)` (O(1))
- Dangerous `loss = 0.01` hardcoded fallback removed from trainer

---

## [1.0.0] — May 2026

### Initial Release

Seven core modules published:

- `zp.memory.OtuxStore` — Selective context-aware memory with 3D semantic coordinates and importance gating
- `zp.stabilize.GradShield` — Real-time gradient health monitor with PyTorch backward hooks
- `zp.stabilize.StabilityCore` — Loss EMA, plateau detection, curvature classification, LR signal
- `zp.stabilize.ModelStabilizer` — Unified stabilization API
- `zp.monitor.NipGraph` — Parity-aware training anomaly detection (x_M/x_W/Y_M/Y_W tracks)
- `zp.weights.WeightVault` — Performance-gated weight snapshots
- `zp.weights.WeightSurgeon` — Multi-model best weight selection via Fisher Information
- `zp.weights.WeightGuard` — EWC-style catastrophic forgetting prevention
- `zp.compute.SafeMath` — NaN-safe loss computation with rational number fallback
- `zp.compat.ZPowerModel` — Transparent PyTorch wrapper
- `zp.compat.augment()` — HuggingFace model augmentation
- `zp.Trainer` — Drop-in intelligent training loop
- `zp.attach()` — One-line model augmentation
