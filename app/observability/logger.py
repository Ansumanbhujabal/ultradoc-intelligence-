"""Structured JSON logger."""

import logging
import json
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry)


def _get_file_handler() -> RotatingFileHandler | None:
    """Create a RotatingFileHandler using the configured log file path."""
    try:
        from app.config import settings

        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(JSONFormatter())
        return handler
    except Exception:
        # If config isn't available or dir creation fails, skip file logging
        return None


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Console handler (existing behaviour)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(JSONFormatter())
        logger.addHandler(stream_handler)

        # Rotating file handler
        file_handler = _get_file_handler()
        if file_handler is not None:
            logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
    return logger
