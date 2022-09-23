import logging
from pathlib import Path
from typing import TextIO

from config import cfg

log_formatter = logging.Formatter(fmt="%(levelname)s: %(message)s")

log = logging.getLogger(Path(__file__).stem)
log.setLevel(cfg.log.level)
stdout_handler = logging.StreamHandler()
stdout_handler.setFormatter(log_formatter)
log.addHandler(stdout_handler)
log.debug("Logger setup finished")


def log_add_stream_handlers(
    logger: logging.Logger, additional_streams: list[TextIO] = []
):
    for stream in additional_streams:
        logger.addHandler(logging.StreamHandler(stream))
    return log
