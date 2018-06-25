from enum import Enum


class PriorsModes(Enum):
    INFORMATIVE = "informative"
    NONINFORMATIVE = "noninformative"


class Target_Data_Type(Enum):
    EMPIRICAL = "empirical"
    SYNTHETIC = "synthetic"