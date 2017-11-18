# Logs and errors

import logging
import os
import warnings

from tvb_epilepsy.base.constants.configurations import FOLDER_LOGS


def initialize_logger(name, target_folder=FOLDER_LOGS):
    """
    create logger for a given module
    :param name: Logger Base Name
    :param target_folder: Folder where log files will be written
    """
    if not (os.path.isdir(target_folder)):
        os.mkdir(target_folder)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(target_folder, name + '.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    return logger


def warning(msg, logger=None, print_warning=True):
    msg = "\n\n" + msg  + "\n"
    if logger is not None:
        logger.warning(msg)
    if print_warning:
        warnings.warn(msg)


def raise_value_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nValueError: " + msg + "\n")
    raise ValueError(msg)


def raise_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nError: " + msg + "\n")
    raise


def raise_import_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nImportError: " + msg + "\n")
    raise ImportError(msg)


def raise_not_implemented_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nNotImplementedError: " + msg + "\n")
    raise NotImplementedError(msg)