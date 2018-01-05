"""
Various configurations which might or might not be system based, should be specified here.
"""

TIME_DELAYS_FLAG = 0.0

ADDITIVE_NOISE = "Additive"
MULTIPLICATIVE_NOISE = "Multiplicative"
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
HDF5_LIB = "libjhdf5.dylib"
LIB_PATH = "/Applications/Episense.app/Contents/Java"
JAR_PATH = "/Applications/Episense.app/Contents/Java/episense-fx-app.jar"
JAVA_MAIN_SIM = "de.codebox.episense.fx.StartSimulation"

import numpy as np
MIN_SINGLE_VALUE = np.finfo("single").min
MAX_SINGLE_VALUE = np.finfo("single").max
MAX_INT_VALUE = np.iinfo(np.int64).max
MIN_INT_VALUE = np.iinfo(np.int64).max


