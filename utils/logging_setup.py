from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging records to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno

        # Find the first frame outside of logging to get correct caller info
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    app_name: str = "necthrall",
    log_dir: str = "logs",
    level: Optional[str] = None,
    development: Optional[bool] = None,
) -> None:
    """Configure Loguru and intercept stdlib logging.

    This creates a file sink (rotating) with JSON serialization and a console sink
    for development. It also installs an InterceptHandler so third-party libraries
    that use the stdlib logging API are forwarded to Loguru.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()

    if development is None:
        development = os.environ.get("ENV", "production").lower() == "development"

    logs_path = Path(log_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    # Remove existing Loguru handlers to avoid duplicate output
    logger.remove()

    # Console sink (readable) for local development/debugging
    logger.add(sys.stdout, level=level, format="{message}", enqueue=True, catch=True)

    # File sink: JSON-serialized, rotated by size
    logger.add(
        str(logs_path / f"{app_name}_{{time}}.log"),
        rotation="10 MB",
        retention="14 days",
        level=level,
        format="{message}",
        serialize=True,
        enqueue=True,
        catch=True,
    )

    # Install intercept handler for stdlib logging
    intercept_handler = InterceptHandler()
    logging.root.handlers = [intercept_handler]
    logging.basicConfig(handlers=[intercept_handler], level=level)


__all__ = ["setup_logging", "InterceptHandler"]
