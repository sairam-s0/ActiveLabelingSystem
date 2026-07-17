# src/core/logger.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Base logs directory
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
if LOGS_DIR.exists() and not LOGS_DIR.is_dir():
    LOGS_DIR = Path(__file__).resolve().parents[2] / "logs_dir"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "als.log"

def setup_logger():
    logger = logging.getLogger("ALS")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File handler with rotation (5MB files, keep last 3)
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"[Logger] Failed to initialize file logging: {e}")

    logger.info("Logging initialized. Writing logs to %s", LOG_FILE)
    return logger

logger = setup_logger()
