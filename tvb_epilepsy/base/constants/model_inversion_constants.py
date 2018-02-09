
from tvb_epilepsy.base.constants.model_constants import X1_DEF, X1_EQ_CR_DEF

# Model inversion constants
X1EQ_MIN = -2.0
X1EQ_MAX = X1_EQ_CR_DEF
X1_REST = X1_DEF
X1_EQ_CR = X1_EQ_CR_DEF
TAU1_DEF = 0.5
TAU1_MIN = 0.1
TAU1_MAX = 1.0
TAU0_DEF = 10.0
TAU0_MIN = 3.0
TAU0_MAX = 30.0
K_MIN = 0.0
K_MAX = 3.0
MC_MIN = 0.0
MC_MAX = 1.0
MC_MAX_MIN_RATIO = 1000.0

# ODE model inversion constants
X1INIT_MIN = -2.0
X1INIT_MAX = 0.0
ZINIT_MIN = 1.0
ZINIT_MAX = 5.0

# SDE model inversion constants
SIG_DEF = 10 ** -2
X1_MIN = -2.0
X1_MAX = 1.0
Z_MIN = 0.0
Z_MAX = 6.0

# Statistical model constants
SIG_EQ_DEF = (X1_EQ_CR_DEF-X1_DEF)/10

# ODE statistical model constants
# EULER_METHODS = ["backward", "midpoint", "forward"]
# OBSERVATION_MODEL_EXPRESSIONS=["lfp", "x1_offset", "x1z_offset"]
# OBSERVATION_EXPRESSION_DEF = "lfp"
OBSERVATION_MODELS=[ "seeg_logpower", "seeg_power", "lfp_power"]
OBSERVATION_MODEL_DEF = "seeg_logpower"
SIG_INIT_DEF = SIG_EQ_DEF

