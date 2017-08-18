"""
Various configurations which might or might not be system based, should be specified here.
"""

import os
import platform
from datetime import datetime

USER_HOME = os.path.expanduser("~")

FOLDER_VEP_ONLINE = os.path.join(USER_HOME, 'Dropbox', 'Work', 'VBtech', 'DenisVEP', 'Results')
FOLDER_VEP = FOLDER_VEP_ONLINE

if platform.node()=='dionperdMBP':
    FOLDER_VEP_HOME = os.path.join(USER_HOME, 'CBR', 'VEP')
    # DATA_CUSTOM = os.path.join(USER_HOME, 'CBR', 'svn', 'episense', 'demo-data')
    DATA_TVB = os.path.join(USER_HOME, 'CBR', 'svn', 'tvb', 'tvb-data', 'tvb-data')
    # DATA_CUSTOM = os.path.join(USER_HOME, 'Dropbox/Work/VBtech/DenisVEP/Results/PATI_HH')
    # DATA_CUSTOM = os.path.join(USER_HOME, 'Dropbox/Work/VBtech/DenisVEP/JUNCH')
    DATA_CUSTOM = os.path.join(FOLDER_VEP, 'CC/TVB1')

else:
    FOLDER_VEP_HOME = os.path.join(USER_HOME, 'VEP')
    # DATA_CUSTOM = os.path.join(USER_HOME, 'CBR_software', 'svn-episense', 'demo-data')
    DATA_TVB = os.path.join(USER_HOME, 'CBR_software', 'svn-tvb', 'tvb-data', 'tvb-data')
    # DATA_CUSTOM = os.path.join(USER_HOME, 'Dropbox/Work/VBtech/DenisVEP/Results/PATI_HH')
    # DATA_CUSTOM = os.path.join(USER_HOME, 'Dropbox/Work/VBtech/DenisVEP/JUNCH')
    DATA_CUSTOM = os.path.join(FOLDER_VEP, 'CC/TVB1')

if not (os.path.isdir(FOLDER_VEP)):
    os.mkdir(FOLDER_VEP)

# Folder where input data will be
# FOLDER_DATA = os.path.join(FOLDER_VEP, 'data')

# Folder where logs will be written
FOLDER_LOGS = os.path.join(FOLDER_VEP_HOME, 'logs'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))

# Folder where results will be saved
FOLDER_RES = os.path.join(FOLDER_VEP_HOME, 'results'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
if not (os.path.isdir(FOLDER_RES)):
        os.mkdir(FOLDER_RES)
# Figures related settings:
VERY_LARGE_SIZE = (40, 20)
LARGE_SIZE = (20, 15)
SMALL_SIZE = (15, 10)
FIG_SIZE = SMALL_SIZE
FOLDER_FIGURES = os.path.join(FOLDER_VEP_HOME, 'figures'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
if not (os.path.isdir(FOLDER_FIGURES)):
        os.mkdir(FOLDER_FIGURES)
FIG_FORMAT = 'png'
SAVE_FLAG = True
SHOW_FLAG = False
MOUSEHOOVER = False

# Default model parameters
X0_DEF = 0.0
X0_CR_DEF = 1.0
E_DEF = 0.0
A_DEF = 1.0
B_DEF = -2.0
K_DEF = 10.0
I_EXT1_DEF = 3.1
YC_DEF = 1.0
X1_DEF = -5.0 / 3.0
X1_EQ_CR_DEF = -4.0 / 3.0
ADDITIVE_NOISE="Additive"
MULTIPLICATIVE_NOISE="Multiplicative"
MAX_DISEASE_VALUE = 1.0 - 10 ** -3

# Simulation and data read folder amd flags:
MODEL = '6v'
CUSTOM = 'custom'
TVB = 'tvb'
SIMULATION_MODE = TVB
DATA_MODE = CUSTOM

# Normalization configuration
WEIGHTS_NORM_PERCENT = 95

NOISE_SEED = 42

SYMBOLIC_CALCULATIONS_FLAG = False

# Options: "auto_eigenvals",  "auto_disease", "auto_epileptogenicity", "auto_excitability",
# or "user_defined", in which case we expect a number equal to from 1 to hypothesis.n_regions
EIGENVECTORS_NUMBER_SELECTION = "auto_eigenvals"
WEIGHTED_EIGENVECTOR_SUM = True
INTERACTIVE_ELBOW_POINT = False


# Information needed for the custom simulation
HDF5_LIB="libjhdf5.dylib"
LIB_PATH="/Applications/Episense.app/Contents/Java"
JAR_PATH="/Applications/Episense.app/Contents/Java/episense-fx-app.jar"
JAVA_MAIN_SIM="de.codebox.episense.fx.StartSimulation"

VOIS = {
    "CustomEpileptor": ['x1', 'z', 'x2'],
    "Epileptor": ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp'],
    "EpileptorDP": ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp'],
    "EpileptorDPrealistic": ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'x0ts', 'slopeTS', 'Iext1ts', 'Iext2ts', 'Kts', 'lfp'],
    "EpileptorDP2D": ['x1', 'z']
}