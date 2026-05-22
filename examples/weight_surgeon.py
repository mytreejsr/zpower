"""
ZPower Example 3 — Select best weights from multiple models
"""
import numpy as np
from zpower.weights import WeightSurgeon

# Option A: Manually add sources
surgeon = WeightSurgeon(conflict_resolution="highest_performer")
surgeon.add_source({"layer1": np.random.randn(4,4).astype("float32")},
                   label="run_jan", perf_score=0.72)
surgeon.add_source({"layer1": np.random.randn(4,4).astype("float32")},
                   label="run_mar", perf_score=0.89)

best = surgeon.select_best()
print("Selection report:", surgeon.selection_report())

# Option B: Auto-discover from folder (interactive)
# surgeon2 = WeightSurgeon()
# surgeon2.auto_discover("./checkpoints/")   # asks y/N per file
# best2 = surgeon2.select_best()
