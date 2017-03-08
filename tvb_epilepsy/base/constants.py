"""
Various configurations witch might or might not be system based, should be specified here.
"""

import os
import platform
from datetime import datetime

USER_HOME = os.path.expanduser("~")

if platform.node()=='dionperdMBP':
    FOLDER_VEP = os.path.join(USER_HOME, 'CBR','VEP')
    DATA_CUSTOM = os.path.join(USER_HOME, 'CBR', 'svn', 'episense', 'demo-data')
    DATA_TVB = os.path.join(USER_HOME, 'CBR','svn','tvb', 'tvb-data', 'tvb-data')
else:
    FOLDER_VEP = os.path.join(USER_HOME, 'VEP')
    DATA_CUSTOM = os.path.join(USER_HOME, 'CBR_software', 'svn-episense', 'demo-data')
    DATA_TVB = os.path.join(USER_HOME,'CBR_software', 'svn-tvb', 'tvb-data', 'tvb-data')

if not (os.path.isdir(FOLDER_VEP)):
    os.mkdir(FOLDER_VEP)

# Folder where input data will be 
FOLDER_DATA = os.path.join(FOLDER_VEP, 'data')

# Folder where logs will be written
FOLDER_LOGS = os.path.join(FOLDER_VEP, 'logs'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')  )

# Folder where results will be saved
FOLDER_RES = os.path.join(FOLDER_VEP, 'results'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M') ) 
if not (os.path.isdir(FOLDER_RES)):
        os.mkdir(FOLDER_RES)
# Figures related settings:
VERY_LARGE_SIZE = (30, 15)
LARGE_SIZE = (20, 15)
SMALL_SIZE = (15, 10)
FOLDER_FIGURES = os.path.join(FOLDER_VEP, 'figures'+datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M') ) 
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
K_DEF = 1.0
I_EXT1_DEF = 3.1
YC_DEF = 1.0
X1_DEF = -5.0 / 3.0
X1_EQ_CR_DEF = -4.0 / 3.0

# Simulation and data read folder amd flags:
MODEL = '6v'
SIMULATION_MODE = 'tvb'  # 'ep' or 'tvb
DATA_MODE = 'ep'  # 'ep' or 'tvb'

# Normalization configuration:
WEIGHTS_NORM_PERCENT = 95

NOISE_SEED = 42

#Attributes to be read or written for hypothesis object and files:
hyp_attributes_dict = {"Hypothesis name": "name", "Model Epileptogenicity": "E", "Pathological Excitability": "x0",
                       "LSA Propagation Strength": "lsa_ps", "x1 Equilibria": "x1EQ",
                       "z Equilibria": "zEQ", "Afferent coupling at equilibrium": "Ceq",
                       "Connectivity": "weights", "Permittivity Coupling": "K", "Iext1": "Iext1",
                       "yc": "yc", "Critical x0": "x0cr", "x0 scaling": "rx0", "EZ hypothesis": "seizure_indices",
                       "x1EQcr": "x1EQcr", "x1LIN": "x1LIN", "x1SQ": "x1SQ",
                       "lsa_eigvals": "lsa_eigvals", "lsa_eigvects": "lsa_eigvects", "lsa_ps_tot": "lsa_ps_tot"}

