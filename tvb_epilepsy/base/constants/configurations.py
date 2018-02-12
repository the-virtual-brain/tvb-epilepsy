# coding=utf-8

import os
import platform
import tvb_epilepsy
from datetime import datetime


# Generic configurations
##################################################
user_home = os.path.expanduser("~")
WORK_FOLDER = os.getcwd()
module_path = os.path.dirname(tvb_epilepsy.__file__)

STATS_MODELS_PATH = os.path.join(module_path, "service", "model_inversion", "stan", "models")
CMDSTAN_PATH = os.path.join(user_home, "ScientificSoftware/git/cmdstan")


# IN data
##################################################
IN_TEST_DATA = os.path.join(WORK_FOLDER, "data")
IN_HEAD = os.path.join(IN_TEST_DATA, "head2")
try:
    import tvb_data
    IN_TVB_DATA = os.path.dirname(tvb_data.__file__)
except ImportError:
    IN_TVB_DATA = WORK_FOLDER

# Overwrite for Dionysios's env
if 'dionperd' in platform.node():
    WORK_FOLDER = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', 'results')
    IN_HEAD = os.path.join(WORK_FOLDER, "CC", 'TVB3', 'Head')


# OUT folders for Logs, Results or Figures
##################################################
separate_results_by_run = False     # Set TRUE, when you want logs/results/figures to be in different files / each run
FOLDER_LOGS = os.path.join(WORK_FOLDER, "vep_out", "logs")
FOLDER_RES = os.path.join(WORK_FOLDER, "vep_out", "res")
FOLDER_FIGURES = os.path.join(WORK_FOLDER, "vep_out", "figs")

if separate_results_by_run:
    FOLDER_LOGS = FOLDER_LOGS + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
    FOLDER_RES = FOLDER_RES + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
    FOLDER_FIGURES = FOLDER_FIGURES + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')

if not (os.path.isdir(FOLDER_LOGS)):
    os.makedirs(FOLDER_LOGS)
if not (os.path.isdir(FOLDER_RES)):
    os.makedirs(FOLDER_RES)
if not (os.path.isdir(FOLDER_FIGURES)):
    os.makedirs(FOLDER_FIGURES)


# Print / Figures related configurations
##################################################
VERY_LARGE_SIZE = (40, 20)
VERY_LARGE_PORTRAIT = (30, 50)
SUPER_LARGE_SIZE = (80, 40)
LARGE_SIZE = (20, 15)
SMALL_SIZE = (15, 10)
FIG_SIZE = SMALL_SIZE
FIG_FORMAT = 'png'
SAVE_FLAG = True
SHOW_FLAG = False
MOUSE_HOOVER = False
