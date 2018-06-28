# coding=utf-8

from tvb_fit.base.config import GenericConfig, InputConfig, OutputConfig, FiguresConfig
from tvb_fit.base.config import CalculusConfig as CalculusConfigBase


class SimulatorConfig(object):
    USE_TIME_DELAYS_FLAG = True
    MODE = GenericConfig.MODE_TVB


class HypothesisConfig(object):

    def __init__(self, head_folder=None):
        self.head_folder = head_folder


class CalculusConfig(CalculusConfigBase):
    SYMBOLIC_CALCULATIONS_FLAG = False
    # Options: "auto_eigenvals",  "auto_disease", "auto_epileptogenicity", "auto_excitability",
    # or "user_defined", in which case we expect a number equal to from 1 to hypothesis.n_regions
    LSA_METHOD = "1D"  # other options: "2D", "auto"
    EIGENVECTORS_NUMBER_SELECTION = "auto_eigenvals"
    WEIGHTED_EIGENVECTOR_SUM = True


class Config(object):
    generic = GenericConfig()
    figures = FiguresConfig()
    calcul = CalculusConfig()
    simulator = SimulatorConfig()

    def __init__(self, head_folder=None, data_mode=GenericConfig.MODE_JAVA,
                 raw_data_folder=None,
                 output_base=None, separate_by_run=False):
        self.input = InputConfig(head_folder, data_mode, raw_data_folder)
        self.out = OutputConfig(output_base, separate_by_run)
        self.hypothesis = HypothesisConfig(head_folder)
