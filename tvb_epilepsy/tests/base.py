# -*- coding: utf-8 -*-

import os
from tvb_epilepsy.base.constants.config import Config


class BaseTest(object):
    config = Config()

    @classmethod
    def setup_class(cls):
        for direc in (cls.config.out.FOLDER_LOGS, cls.config.out.FOLDER_RES, cls.config.out.FOLDER_FIGURES,
                      cls.config.out.FOLDER_TEMP):
            if not os.path.exists(direc):
                os.makedirs(direc)

    @classmethod
    def teardown_class(cls):
        for direc in (cls.config.out.FOLDER_LOGS, cls.config.out.FOLDER_RES, cls.config.out.FOLDER_FIGURES,
                      cls.config.out.FOLDER_TEMP):
            for dir_file in os.listdir(direc):
                os.remove(os.path.join(os.path.abspath(direc), dir_file))
            os.removedirs(direc)
