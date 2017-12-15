

import sys
import os
import subprocess
import time


# TODO: threading:
# https://docs.python.org/3/library/threading.html
# https://stackoverflow.com/questions/984941/python-subprocess-popen-from-a-thread


def execute_command(command, cwd=os.getcwd(), shell=True):
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
        print("The process ran for " + str(time.time() - tic))
        raise subprocess.CalledProcessError(exitCode, command)
