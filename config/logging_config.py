"""
Centralized logging configuration.

Logs to both console (formatted) and a rotating file. Every module should use:

    from config.logging_config import get_logger
    logger = get_logger(__name__)
"""

import logging
import logging.handlers
import sys

from config.config import settings


_LOGGERS_CONFIGURED: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger configured once per process."""
    logger = logging.getLogger(name)

    if name in _LOGGERS_CONFIGURED:
        return logger

    logger.setLevel(settings.log_level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (rotating, 10MB x 5 files)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _LOGGERS_CONFIGURED.add(name)
    return logger
