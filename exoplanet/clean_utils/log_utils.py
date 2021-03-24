from config import *
import logging


def get_logger(logger_name, filename):
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(
        os.path.join(log_dir, filename), 'w')
    handler.setFormatter(simple_formatter)
    handler.setLevel(logging.WARN)
    logger.addHandler(handler)
    return logger
