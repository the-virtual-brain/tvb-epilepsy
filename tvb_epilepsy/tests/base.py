# -*- coding: utf-8 -*-

import os
import shutil

here = os.path.dirname(os.path.abspath(__file__))
temporary_folder = 'temp'


def get_temporary_folder():
    if not os.path.exists(temporary_folder):
        os.makedirs(temporary_folder)
    return temporary_folder


def get_temporary_files_path(*args):
    file_path = os.path.join(get_temporary_folder(), *args)
    return file_path


def remove_temporary_test_files():
    shutil.rmtree(temporary_folder)
