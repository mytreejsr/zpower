# Changelog

## v1.3.0 — May 2026

**Theme: Hardening + Smart Features + Developer Experience**

### Critical Bug Fixes

1. **`_trainer.py`: `self._ema_last(model)` crash** — `_ema_last` is a module-level function, not a method. Using `self._ema_last()` caused `AttributeError` at runtime when StabilityCore fallback was triggered. Fixed to `_ema_last(model)`.

2. **`test_otux.py`: `query_by_coord()` returns dicts** — Test accessed `deep_results[0].token` but `query_by_coord()` returns dicts since v1.2. Fixed to `deep_results[0]["token"]`.

3. **`test_integration.py`: version assertion outdated** — Still asserted `__version__ == "1.0.0"`. Fixed to `"1.3.0"`.

4. **`__init__.py`: AutoHeal docstring said "Reserved for v2"** — AutoHeal is fully implemented since v1.1. Both `attach()` and `Trainer` docstrings updated with actual description.

5. **`ZPowerModel.forward()`: `_step` never incremented** — Without Trainer, all OtuxStore memory entries were tagged `step_0`. Now `_step` increments in `forward()`, and `zp_on_step_end()` no longer double-increments.

6. **`WeightVault.load()`: security risk with `allow_pickle=True`** — Same RCE vulnerability that was fixed in `WeightSurgeon` for `torch.load()`. Now load path validates file existence and uses allow_pickle only for metrics dict (vault owner-created data).

### Security Fixes

7. **`ZPConfig.validate()`: used `assert` instead of `ValueError`** — Python `assert` statements can be disabled with `-O` flag, silently disabling validation. All checks now raise `ValueError`. Validation also covers ALL fields (was only 4 of 27).

8. **`WeightSurgeon._load_state()`: silent fallback to `weights_only=False`** — Now logs a WARNING via `zplog` when falling back to full pickle deserialization.

9. **`GradShield._remove_hooks()`: `RuntimeError` on already-removed hooks** — Now catches `RuntimeError`/`ValueError` instead of crashing.

### Smart Features

10. **Smart Eviction in OtuxStore** — `_evict()` was O(N) scan over all entries. Now uses a min-heap for O(log N) eviction of lowest-scored entries. Heap validated before use with fallback to O(N) scan on corruption.

11. **Adaptive Warmup in GradShield** — Previously used fixed `_WARMUP_STEPS=20` before activating adaptive thresholds. Now uses variance-based confidence check: exits warmup when coefficient of variation is stable OR after 20 samples. Prevents premature adaptive thresholds on layers with few gradient observations.

12. **Multiple Heal Strategies in AutoHeal** — New `strategy` parameter:
    - `'both'` (default) — rollback + LR cut (v1.1/v1.2 behaviour)
    - `'rollback_only'` — restore weights only, keep LR
    - `'lr_only'` — reduce LR only, no weight rollback
    - `'restart'` — rollback + LR cut + reset optimizer momentum

13. **Overhead Tracking in ZPowerModel** — New `_zp_overhead_ms` and `_zp_forward_count` track zpower's overhead per forward call. Access via `overhead_ms()`, `overhead_per_call_ms()`, or `health_report()`.

14. **`ZPowerModel.pprint_status()`** — Formatted console output of all ZPower component status in a readable dashboard format.

### API Improvements

15. **`__repr__` for all classes** — `GradShield`, `StabilityCore`, `ModelStabilizer`, `NipGraph`, `AutoHeal`, `WeightVault`, `VaultSnapshot`, `WeightGuard`, `WeightSurgeon`, `SafeMath`, `ZPConfig`, `ZPowerModel`, `Trainer`, `ImportanceWeights`, `OtuxEntry` all have useful `__repr__` now.

16. **`AutoHeal.reset()`** — Clears heal state for reuse between training runs.

17. **`WeightVault.get_snapshot_metrics()`** — Retrieve stored metrics (loss, accuracy, etc.) for all snapshots of a layer.

18. **`WeightVault.save()` now persists metrics** — Previously, `VaultSnapshot.metrics` was lost on save/load. Now stored alongside weights in `.npz` file.

19. **`VaultSnapshot.__repr__`** — Shows layer name, epoch, and score for debugging.

20. **`OtuxStore.query_by_coord()` now increments `access_count`** — Was inconsistent with `query()` which increments. Affects eviction priority.

21. **`SafeMath._tokenize()` eviction from pocket_map** — Was O(N) scan to find reverse mapping. Now uses `_pocket_reverse` dict for O(1) eviction.

22. **`NipGraph.VarState.history` is now `deque`** — Was `list` with manual `pop(0)` (O(N)). Now `deque(maxlen=200)` with O(1) append/trim.

23. **`zpower/utils/__init__.py` exports all logging functions** — Now exports `debug`, `info`, `warning`, `error`, `get_level` (was missing).

24. **`zpower/__init__.py` exports `OtuxEntry` and `FrozenToken`** — Were exported by subpackage `__init__.py` but not top-level.

25. **`OtuxStore._prep()` clearer error messages** — Non-numeric input and zero-norm vectors now explain what went wrong and how to fix it.

26. **`OtuxStore` constructor validation** — `dim`, `max_entries`, and threshold ordering validated on construction with clear `ValueError`.

27. **`WeightVault` constructor validation** — `vault_threshold` and `max_per_layer` validated on construction.

28. **`WeightGuard` constructor validation** — `protection_strength` and `adapt_rate` validated on construction.

29. **`SafeMath` constructor validation** — `pocket_capacity` must be > 0.

30. **`zplog.set_level()` validates input** — Invalid level raises `ValueError` instead of silently accepting.

### Test Improvements

31. **AutoHeal unit tests** — 6 new tests: continue on healthy, skip on NaN, strategy validation, valid strategies, reset, max_heals raises.

32. **`__repr__` tests** — All classes with `__repr__` are tested.

33. **Config validation tests** — `ZPConfig.validate()` tested for all invalid inputs.

34. **Health API consistency tests** — All core classes verified to have both `health()` and `status()` returning dicts with "status" key.

35. **Smart eviction tests** — OtuxStore eviction at max_entries verified.

36. **OtuxStore constructor validation tests** — Bad dim, max_entries, thresholds tested.

37. **WeightVault file not found test** — `load()` raises `FileNotFoundError` for missing files.

### Documentation

38. **`pyproject.toml` Homepage URL fixed** — Was `nnbhoi/zpower`, now `mytreejsr/zpower`.

39. **README updated for v1.3.0** — All new features documented.

---

## v1.2.0 — May 2026

**Theme: Critical Bug Fixes + Security + Performance**

3 critical bugs fixed. 5 high-priority correctness issues resolved. 3 performance upgrades. 1 new feature (WeightSurgeon.auto_discover).

### Critical Bug Fixes
- ZPowerModel.parameters() returned empty iterator
- StabilityCore plateau detection compared incompatible values
- write_batch() overwrote the wrong entry result

### High Priority Fixes
- OtuxStore memory was dead code during training
- GradShield health state never reached WeightVault
- torch.load() security vulnerability (RCE risk)
- NipGraph false alerts on converged models
- NipGraph step counter drift

### Performance Upgrades
- OtuxStore.query() O(N log N) to O(N) via argpartition
- GradShield.status() O(N) to O(1) via running counters
- StabilityCore._history deque replaces list + trim
- SafeMath pocket OrderedDict for O(1) LRU eviction

---

## v1.1.0 — May 2026

**Theme: Performance + AutoHeal + Adaptive Thresholds**

- AutoHeal Engine: automatic training failure recovery
- OTUX-S Memory: 100x faster writes, write_batch(), hash() replaces MD5
- GradShield: adaptive per-layer thresholds via Welford algorithm
- Dangerous hardcoded fallback removed

---

## v1.0.0 — May 2026

**Initial release.** Seven core modules: OtuxStore, GradShield, StabilityCore, NipGraph, WeightVault, WeightSurgeon, WeightGuard, ZPowerModel, Trainer.
