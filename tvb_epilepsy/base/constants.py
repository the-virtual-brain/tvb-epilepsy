"""
Various configurations which might or might not be system based, should be specified here.
"""

import os
import platform
from datetime import datetime

USER_HOME = os.path.expanduser("~")

if platform.node()=='dionperdMBP':
    FOLDER_VEP = os.path.join(USER_HOME, 'CBR', 'VEP')
    # DATA_CUSTOM = os.path.join(USER_HOME, 'CBR', 'svn', 'episense', 'demo-data')
    DATA_TVB = os.path.join(USER_HOME, 'CBR', 'svn', 'tvb', 'tvb-data', 'tvb-data')
    # DATA_CUSTOM = os.path.join(USER_HOME, 'Dropbox/Work/VBtech/DenisVEP/Results/PATI_HH')
    # DATA_CUSTOM = os.path.join(USER_HOME, 'Dropbox/Work/VBtech/DenisVEP/JUNCH')
    DATA_CUSTOM = os.path.join(FOLDER_VEP, 'CC/TVB1')

else:
    FOLDER_VEP = os.path.join(USER_HOME, 'VEP')
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
FOLDER_LOGS = os.path.join(FOLDER_VEP, 'logs'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))

# Folder where results will be saved
FOLDER_RES = os.path.join(FOLDER_VEP, 'results'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
if not (os.path.isdir(FOLDER_RES)):
        os.mkdir(FOLDER_RES)
# Figures related settings:
VERY_LARGE_SIZE = (30, 15)
LARGE_SIZE = (20, 15)
SMALL_SIZE = (15, 10)
FOLDER_FIGURES = os.path.join(FOLDER_VEP, 'figures'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'))
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
K_DEF = 10
I_EXT1_DEF = 3.1
YC_DEF = 1.0
X1_DEF = -5.0 / 3.0
X1_EQ_CR_DEF = -4.0 / 3.0
ADDITIVE_NOISE="Additive"
MULTIPLICATIVE_NOISE="Multiplicative"

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

EIGENVECTORS_NUMBER_SELECTION = "auto_eigenvals" # Options: "auto_eigenvals", "auto_epileptogenicity", "auto_x0", a number equal to seizure_indices or from 1 to hypothesis.n_regions
INTERACTIVE_ELBOW_POINT=False


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