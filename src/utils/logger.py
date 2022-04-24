import logging, sys

_LOGGERS = []

def setup_loggers(cfg):
    # prepare loggers
    logger = logging.getLogger("MOT Challenge")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler("{}/log.txt".format(cfg.OUTPUT_DIR))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    _LOGGERS.append(logger)

def get_logger():
    print(len(_LOGGERS))
    return _LOGGERS[-1]