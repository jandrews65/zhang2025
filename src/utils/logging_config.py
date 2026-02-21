"""Logging configuration using loguru.

Provides consistent logging setup across all pipeline scripts.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_dir: Path = Path("logs"),
    level: str = "INFO",
    script_name: str = "pipeline",
) -> None:
    """Configure loguru for the pipeline.

    Sets up both console and file logging with rotation.

    Args:
        log_dir: Directory for log files.
        level: Minimum log level.
        script_name: Name used for the log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # File handler with rotation
    logger.add(
        log_dir / f"{script_name}.log",
        level=level,
        rotation="50 MB",
        retention="30 days",
        compression="gz",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        ),
    )

    logger.info(f"Logging initialized for {script_name}")
