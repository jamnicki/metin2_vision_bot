import sys
import time

from loguru import logger
from itertools import cycle
from pathlib import Path
from typing import NewType, Optional

Success = NewType("Success", bool)


def setup_logger(script_name: str, level: str):
    level = level.upper()
    logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
    logger.add(sys.stdout, level=level)
    log_filename = f"{script_name}__{str(time.time()).replace('.', 'f')}.log"
    log_filepath = Path("logs") / log_filename
    log_filepath.parent.mkdir(exist_ok=True)
    logger.add(log_filepath, level=level)


def channel_generator(min_: int = 1, max_: int = 8, start: Optional[int] = None):
    channels = list(range(min_, max_ + 1))
    if start is not None:
        assert 1 <= start <= 8, f"Start channel must be between 1 and 8, '{start}' given."
        for ch in channels[start - 1:]:
            yield ch
    channels_cycle = cycle(channels)
    while True:
        yield next(channels_cycle)
