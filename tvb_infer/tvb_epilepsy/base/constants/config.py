# coding=utf-8

from tvb_infer.base.config import GenericConfig, InputConfig, OutputConfig, FiguresConfig
from tvb_infer.base.config import CalculusConfig as CalculusConfigBase


class SimulatorConfig(object):
    USE_TIME_DELAYS_FLAG = True
    MODE = GenericConfig.MODE_TVB


class HypothesisConfig(object):

    def __init__(self, head_folder=None):
        self.head_folder = head_folder


class CalculusConfig(CalculusConfigBase):
    SYMBOLIC_CALCULATIONS_FLAG = False


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
