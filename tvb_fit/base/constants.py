from enum import Enum


class PriorsModes(Enum):
    INFORMATIVE = "informative"
    NONINFORMATIVE = "noninformative"


class Target_Data_Type(Enum):
    EMPIRICAL = "empirical"
    SYNTHETIC = "synthetic"


WHITE_NOISE = "White"
COLORED_NOISE = "Colored"
NOISE_SEED = 42

TIME_DELAYS_FLAG = 0