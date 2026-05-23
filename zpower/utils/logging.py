# zpower/utils/logging.py  — v1.3.0
# v1.3: Added get_timestamp() utility
from __future__ import annotations
import sys
from datetime import datetime

_LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
_MIN    = "INFO"


def set_level(level: str):
    """Set minimum log level. One of: DEBUG, INFO, WARNING, ERROR."""
    global _MIN
    level = level.upper()
    if level not in _LEVELS:
        raise ValueError(f"Invalid log level: {level}. Valid: {list(_LEVELS.keys())}")
    _MIN = level


def get_level() -> str:
    """Get current minimum log level."""
    return _MIN


def _log(component: str, level: str, msg: str):
    if _LEVELS.get(level.upper(), 1) >= _LEVELS.get(_MIN, 1):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [zp.{component}] {level.upper()}: {msg}", file=sys.stderr)


def debug(c: str, m: str):   _log(c, "DEBUG",   m)
def info(c: str, m: str):    _log(c, "INFO",    m)
def warning(c: str, m: str): _log(c, "WARNING", m)
def error(c: str, m: str):   _log(c, "ERROR",   m)
