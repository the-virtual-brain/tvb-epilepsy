"""
Various configurations which might or might not be system based, should be specified here.
"""

VERY_LARGE_SIZE = (40, 20)
LARGE_SIZE = (20, 15)
SMALL_SIZE = (15, 10)
FIG_SIZE = SMALL_SIZE
FIG_FORMAT = 'png'
SAVE_FLAG = True
SHOW_FLAG = False
MOUSEHOOVER = False

TIME_DELAYS_FLAG = 0.0

# Default model parameters
X0_DEF = 0.0
X0_CR_DEF = 1.0
E_DEF = 0.0
A_DEF = 1.0
B_DEF = 3.0
D_DEF = 5.0
SLOPE_DEF = 0.0
S_DEF = 6.0
GAMMA_DEF = 0.1
K_DEF = 10.0
I_EXT1_DEF = 3.1
I_EXT2_DEF = 0.45
YC_DEF = 1.0
TAU1_DEF = 1.0
TAU2_DEF = 10.0
TAU0_DEF = 2857.0
X1_DEF = -5.0 / 3.0
X1_EQ_CR_DEF = -4.0 / 3.0
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

VOIS = {
    "CustomEpileptor": ['x1', 'z', 'x2'],
    "Epileptor": ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp'],
    "EpileptorDP": ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp'],
    "EpileptorDPrealistic": ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp', 'x0_t', 'slope_t', 'Iext1_t', 'Iext2_t', 'K_t'],
    "EpileptorDP2D": ['x1', 'z']
}

# Soon also: autoregression_dWt, ode, ode_lsa, ...

STATISTICAL_MODEL_TYPES=["autoregression"]
OBSERVATION_MODEL_EXPRESSIONS=["x1z_offset", "x1_offset", "x1"]
OBSERVATION_MODELS=["logpower", "linear_mixing", "linear"]