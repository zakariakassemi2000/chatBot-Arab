# -*- coding: utf-8 -*-
"""
SHIFA AI — Production Logging System

Structured, file-backed logging for all modules.
Console output is minimal; detailed logs go to file.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Model loaded successfully")
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Configuration ──
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Log directory: always create a 'logs/' folder at project root
_PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = os.environ.get("LOG_DIR", str(_PROJECT_ROOT / "logs"))

# File rotation settings
MAX_LOG_SIZE = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT = 5                   # Keep 5 rotated backups

# ── Internal state ──
_handlers_configured = False


def _configure_root_handlers():
    """Configure root-level handlers once (file + console)."""
    global _handlers_configured
    if _handlers_configured:
        return
    _handlers_configured = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # ── Console handler: WARNING+ only (keep terminal clean) ──
    console = logging.StreamHandler(sys.stdout)
    console_level = logging.WARNING if LOG_LEVEL != "DEBUG" else logging.DEBUG
    console.setLevel(console_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ── File handler: always active, DEBUG level ──
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, "shifa_ai.log"),
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # If file logging fails (read-only FS, Docker, etc.), continue silently
        pass


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a module.

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Model initialized")
        logger.warning("Fallback model used")
        logger.error("Model load failed", exc_info=True)
    """
    _configure_root_handlers()
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger
