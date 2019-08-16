import os

from tvb_config.config import Config as ConfigBase
from tvb_config.config import GenericConfig as GenericConfigBase
from tvb_config.config import CalculusConfig, FiguresConfig, InputConfig, OutputConfig


class GenericConfig(GenericConfigBase):
    _module_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    PROBLSTC_MODELS_PATH = os.path.join(_module_path, "samplers", "stan", "models")
    CMDSTAN_PATH = os.path.join(os.path.expanduser("~"), "ScientificSoftware/git/cmdstan")
    MODEL_COMPARISON_PATH = os.path.join(os.path.dirname(_module_path), "extern")
    C_COMPILER = "clang++"

    # Information needed for the Java simulation
    HDF5_LIB = "libjhdf5.dylib"
    LIB_PATH = "/Applications/Episense.app/Contents/Java"
    JAR_PATH = "/Applications/Episense.app/Contents/Java/episense-fx-app.jar"
    JAVA_MAIN_SIM = "de.codebox.episense.fx.StartSimulation"

    # Identify and choose the Simulator, or data folder type to read.
    MODE_JAVA = "java"
    MODE_TVB = "tvb"


class Config(ConfigBase):
    generic = GenericConfig()
    figures = FiguresConfig()
    calcul = CalculusConfig()

    def __init__(self, head_folder=None, data_mode=GenericConfig.MODE_JAVA,
                 raw_data_folder=None,
                 output_base=None, separate_by_run=False):
        self.input = InputConfig(head_folder, data_mode, raw_data_folder)
        self.out = OutputConfig(output_base, separate_by_run)

