# coding=utf-8

import os
import platform
import numpy as np
import tvb_epilepsy
from datetime import datetime


# Generic configurations
##################################################
module_path = os.path.dirname(tvb_epilepsy.__file__)
STATS_MODELS_PATH = os.path.join(module_path, "service", "model_inversion", "stan", "models")
CMDSTAN_PATH = os.path.join(os.path.expanduser("~"), "ScientificSoftware/git/cmdstan")

# Information needed for the Java simulation
HDF5_LIB = "libjhdf5.dylib"
LIB_PATH = "/Applications/Episense.app/Contents/Java"
JAR_PATH = "/Applications/Episense.app/Contents/Java/episense-fx-app.jar"
JAVA_MAIN_SIM = "de.codebox.episense.fx.StartSimulation"

# Identify and choose the Simulator to use data folder from where to read.
JAVA = 'java'
TVB = 'tvb'
DATA_MODE = JAVA


# IN data
##################################################
WORK_FOLDER = os.getcwd()
IN_HEAD = os.path.join(WORK_FOLDER, "data", "head")
try:
    import tvb_data
    IN_TVB_DATA = os.path.dirname(tvb_data.__file__)
except ImportError:
    IN_TVB_DATA = WORK_FOLDER

# Overwrite for Dionysios's env
if 'dionperd' in platform.node():
    WORK_FOLDER = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', 'results')
    IN_HEAD = os.path.join(WORK_FOLDER, "CC", 'TVB3', 'Head')


# OUT folders for Logs, Results or Figures
##################################################
separate_results_by_run = False  # Set TRUE, when you want logs/results/figures to be in different files / each run
FOLDER_LOGS = os.path.join(WORK_FOLDER, "vep_out", "logs")
FOLDER_RES = os.path.join(WORK_FOLDER, "vep_out", "res")
FOLDER_FIGURES = os.path.join(WORK_FOLDER, "vep_out", "figs")
FOLDER_TEMP = os.path.join(WORK_FOLDER, "vep_out", "temp")

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


# Calculus Related settings
##################################################
SYMBOLIC_CALCULATIONS_FLAG = False

# Normalization configuration
WEIGHTS_NORM_PERCENT = 95

# Options: "auto_eigenvals",  "auto_disease", "auto_epileptogenicity", "auto_excitability",
# or "user_defined", in which case we expect a number equal to from 1 to hypothesis.n_regions
EIGENVECTORS_NUMBER_SELECTION = "auto_eigenvals"
WEIGHTED_EIGENVECTOR_SUM = True
INTERACTIVE_ELBOW_POINT = False

MIN_SINGLE_VALUE = np.finfo("single").min
MAX_SINGLE_VALUE = np.finfo("single").max
MAX_INT_VALUE = np.iinfo(np.int64).max
MIN_INT_VALUE = np.iinfo(np.int64).max
