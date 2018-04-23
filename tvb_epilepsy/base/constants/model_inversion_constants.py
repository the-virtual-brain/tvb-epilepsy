from enum import Enum

from tvb_epilepsy.base.constants.model_constants import X1_DEF, X1EQ_CR_DEF

# Model inversion constants
class PriorsModes(Enum):
    INFORMATIVE = "informative"
    NONINFORMATIVE = "noninformative"


class XModes(Enum):
    X0MODE = "x0"
    X1EQMODE = "x1eq"

X1_REST = X1_DEF
X1EQ_CR = X1EQ_CR_DEF
X1EQ_MIN = -1.8
X1EQ_MAX = -1.0  # X1EQ_CR_DEF
X1EQ_DEF = X1_DEF
X1EQ_SCALE = 3
SIG_EQ_DEF = 0.15*(X1EQ_MAX - X1EQ_MIN)
SIGMA_EQ_DEF = SIG_EQ_DEF

X0_MIN = -4.0
X0_MAX = 1.0
X0_DEF = -2.5
X0_SCALE = 3
SIG_X0_DEF = 0.1*(X0_MAX - X0_MIN)
SIGMA_X0_DEF = SIG_X0_DEF

TAU1_DEF = 0.5
TAU1_MIN = 0.1
TAU1_MAX = 1.0
TAU1_SCALE = 6

TAU0_DEF = 10.0
TAU0_MIN = 3.0
TAU0_MAX = 30.0
TAU0_SCALE = 6

K_MIN = 0.0
K_MAX = 10.0
K_SCALE = 6

MC_MIN = 0.0
MC_MAX = 2.0
MC_MAX_MIN_RATIO = 1000.0
MC_SCALE = 6.0

# ODE model inversion constants
X1INIT_MIN = -2.0
X1INIT_MAX = 0.0
ZINIT_MIN = 2.0
ZINIT_MAX = 5.0
DT_DEF = 0.1
SIG_INIT_DEF = 0.1*SIG_EQ_DEF
SIGMA_INIT_DEF = SIG_INIT_DEF
EPS_DEF = 0.1
EPSILON_DEF = EPS_DEF
SCALE_SIGNAL_DEF = 1.0
OFFSET_SIGNAL_DEF = 0.0

class OBSERVATION_MODELS(Enum):
    SEEG_LOGPOWER = 0
    SEEG_POWER = 1
    SOURCE_POWER = 2
    SEEG = [0, 1]

OBSERVATION_MODEL_DEF = "seeg_logpower"

# SDE model inversion constants
class SDE_MODES(Enum):
    CENTERED = "centered"
    NONCENTERED = "noncentered"

SIG_DEF = 0.05
SIGMA_DEF = SIG_DEF
X1_MIN = -2.0
X1_MAX = 1.0
Z_MIN = 0.0
Z_MAX = 6.0

WIN_LEN_RATIO = 10
LOW_FREQ = 10.0
HIGH_FREQ = 256.0
BIPOLAR = False

