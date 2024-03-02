import sys
import time

from loguru import logger
from itertools import cycle
from pathlib import Path


def setup_logger(script_name: str, level: str):
    level = level.upper()
    logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
    logger.add(sys.stdout, level=level)
    log_filename = f"{script_name}__{str(time.time()).replace('.', 'f')}.log"
    log_filepath = Path("logs") / log_filename
    log_filepath.parent.mkdir(exist_ok=True)
    logger.add(log_filepath, level=level)


def channel_generator(min_=1, max_=8):
    channels_cycle = cycle(list(range(min_, max_ + 1)))
    while True:
        yield next(channels_cycle)
