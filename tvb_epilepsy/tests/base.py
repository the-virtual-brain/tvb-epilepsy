# -*- coding: utf-8 -*-

import os
import shutil

from tvb_epilepsy.base.constants.configurations import FOLDER_TEMP


def get_temporary_folder():
    if not os.path.exists(FOLDER_TEMP):
        os.makedirs(FOLDER_TEMP)
    return FOLDER_TEMP


def get_temporary_files_path(*args):
    file_path = os.path.join(get_temporary_folder(), *args)
    return file_path


def remove_temporary_test_files():
    shutil.rmtree(FOLDER_TEMP)
