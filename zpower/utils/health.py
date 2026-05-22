# zpower/utils/health.py
from __future__ import annotations
import time
from typing import Any, Dict


def export_health(components: Dict[str, Any]) -> Dict:
    report: Dict = {"timestamp": time.time(), "components": {}}
    for name, comp in components.items():
        if hasattr(comp, "health"):
            report["components"][name] = comp.health()
        elif hasattr(comp, "status"):
            report["components"][name] = comp.status()
        else:
            report["components"][name] = {"status": "ok"}
    return report
