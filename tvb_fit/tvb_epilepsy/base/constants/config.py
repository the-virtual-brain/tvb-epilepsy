# coding=utf-8

from tvb_fit.base.config import Config as ConfigBase
from tvb_fit.base.config import GenericConfig as GenericConfigBase
from tvb_fit.base.config import CalculusConfig as CalculusConfigBase
from tvb_fit.base.config import InputConfig, OutputConfig, FiguresConfig


class GenericConfig(GenericConfigBase):
    # Information needed for the Java simulation
    HDF5_LIB = "libjhdf5.dylib"
    LIB_PATH = "/Applications/Episense.app/Contents/Java"
    JAR_PATH = "/Applications/Episense.app/Contents/Java/episense-fx-app.jar"
    JAVA_MAIN_SIM = "de.codebox.episense.fx.StartSimulation"


class SimulatorConfig(object):
    USE_TIME_DELAYS_FLAG = True
    MODE_JAVA = "java"


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


class Config(ConfigBase):
    generic = GenericConfig()
    calcul = CalculusConfig()
    simulator = SimulatorConfig()

    def __init__(self, head_folder=None, data_mode=GenericConfig.MODE_TVB,
                 raw_data_folder=None, output_base=None, separate_by_run=False):
        super(Config, self).__init__(head_folder, data_mode,
                                     raw_data_folder, output_base, separate_by_run)
        self.hypothesis = HypothesisConfig(head_folder)
