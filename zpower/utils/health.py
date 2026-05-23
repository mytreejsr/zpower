# zpower/utils/health.py  — v1.3.0
# v1.3: Added component type info in report
from __future__ import annotations
import time
from typing import Any, Dict


def export_health(components: Dict[str, Any]) -> Dict:
    """Aggregate health/status from all ZPower components into a unified report."""
    report: Dict = {"timestamp": time.time(), "components": {}}
    for name, comp in components.items():
        if hasattr(comp, "health"):
            report["components"][name] = comp.health()
        elif hasattr(comp, "status"):
            report["components"][name] = comp.status()
        else:
            report["components"][name] = {"status": "ok"}
        # Add component type info
        report["components"][name]["_type"] = type(comp).__name__
    return report
