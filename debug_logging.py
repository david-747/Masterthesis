"""
Lightweight, structured logging helpers for your CMAB/LP/Simulator stack.

Usage:
    from debug_logging import get_logger, jlog, set_verbosity

    log = get_logger("cmab")
    jlog(log, "event_name", key="value", nested={"ok": True})

    set_verbosity("DEBUG")  # or "INFO", "WARNING"
"""
import json
import logging
from typing import Optional

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

def get_logger(name: str = "cmab", to_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Default to INFO
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if to_file:
        fh = logging.FileHandler(to_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def set_verbosity(level: str) -> None:
    """Set global verbosity for all created loggers (case-insensitive)."""
    level = _LEVELS.get(level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)

def jlog(logger: logging.Logger, event: str, **data) -> None:
    """JSON-log a single line with an 'event' and arbitrary key/values."""
    try:
        msg = json.dumps({"event": event, **data}, default=str, ensure_ascii=False)
    except TypeError:
        # Fallback if something isn't JSON-serializable
        safe = {k: str(v) for k, v in data.items()}
        msg = json.dumps({"event": event, **safe}, ensure_ascii=False)
    logger.info(msg)