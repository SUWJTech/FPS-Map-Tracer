import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(log_file: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("fps_trajectory")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt))
        fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger
