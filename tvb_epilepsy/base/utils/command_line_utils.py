
import sys
import os
import subprocess
import time
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger


# TODO: threading:
# https://docs.python.org/3/library/threading.html
# https://stackoverflow.com/questions/984941/python-subprocess-popen-from-a-thread


def execute_command(command, cwd=os.getcwd(), shell=True, logger=None):
    if logger is None:
        logger = initialize_logger(__name__)
    logger.info("Running process in directory:\n" + cwd)
    logger.info("Command:\n"+ command)
    # TODO: make logger infor printable to the console!
    print("Running process in directory:\n" + cwd)
    print("Command:\n" + command)
    tic = time.time()
    process = subprocess.Popen(command, shell=shell, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True)
    while True:
        nextline = process.stdout.readline()
        if nextline == '' and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()
    output = process.communicate()[0]
    exitCode = process.returncode
    if exitCode == 0:
        return output, time.time() - tic
    else:
        logger.info("The process ran for " + str(time.time() - tic))
        print("The process ran for " + str(time.time() - tic))
        raise subprocess.CalledProcessError(exitCode, command)
