import sys
import time

from loguru import logger


def set_logger_level(script_name: str, level: str):
    logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
    logger.add(sys.stdout, level=level)
    log_filename = f"{script_name}__{str(time.time()).replace('.', 'f')}.log"
    logger.add("logs/" + log_filename, level=level)
