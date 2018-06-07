# coding=utf-8

import os
import numpy as np
import tvb_epilepsy
from datetime import datetime


class GenericConfig(object):
    _module_path = os.path.dirname(tvb_epilepsy.__file__)
    PROBLSTC_MODELS_PATH = os.path.join(_module_path, "service", "model_inversion", "stan", "models")
    CMDSTAN_PATH = os.path.join(os.path.expanduser("~"), "ScientificSoftware/git/cmdstan") # _precompiled
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


class InputConfig(object):
    _base_input = os.getcwd()

    @property
    def HEAD(self):
        if self._head_folder is not None:
            return self._head_folder

        if not self.IS_TVB_MODE:
            # Expecting to run in the top of tvb_epilepsy GIT repo, with the dummy head
            return os.path.join(self._base_input, "data", "head")

        # or else, try to find tvb_data module
        try:
            import tvb_data
            return os.path.dirname(tvb_data.__file__)
        except ImportError:
            return self._base_input

    @property
    def IS_TVB_MODE(self):
        """Identify and choose the Input data type to use"""
        return self._data_mode == GenericConfig.MODE_TVB

    @property
    def RAW_DATA_FOLDER(self):
        if self._raw_data is not None:
            return self._raw_data

        # Expecting to run in the top of tvb_epilepsy GIT repo, with the dummy head
        return os.path.join(self._base_input, "data", "raw")

    def __init__(self, head_folder=None, data_mode=GenericConfig.MODE_JAVA, raw_folder=None):
        self._head_folder = head_folder
        self._data_mode = data_mode
        self._raw_data = raw_folder


class OutputConfig(object):
    subfolder = None

    def __init__(self, out_base=None, separate_by_run=False):
        """
        :param work_folder: Base folder where logs/figures/results should be kept
        :param separate_by_run: Set TRUE, when you want logs/results/figures to be in different files / each run
        """
        self._out_base = out_base or os.path.join(os.getcwd(), "vep_out")
        self._separate_by_run = separate_by_run

    @property
    def FOLDER_LOGS(self):
        folder = os.path.join(self._out_base, "logs")
        if self._separate_by_run:
            folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        return folder

    @property
    def FOLDER_RES(self):
        folder = os.path.join(self._out_base, "res")
        if self._separate_by_run:
            folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        if self.subfolder is not None:
            os.path.join(folder, self.subfolder)
        return folder

    @property
    def FOLDER_FIGURES(self):
        folder = os.path.join(self._out_base, "figs")
        if self._separate_by_run:
            folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        if not (os.path.isdir(folder)):
            os.makedirs(folder)
        if self.subfolder is not None:
            os.path.join(folder, self.subfolder)
        return folder

    @property
    def FOLDER_TEMP(self):
        return os.path.join(self._out_base, "temp")


class FiguresConfig(object):
    VERY_LARGE_SIZE = (40, 20)
    VERY_LARGE_PORTRAIT = (30, 50)
    SUPER_LARGE_SIZE = (80, 40)
    LARGE_SIZE = (20, 15)
    SMALL_SIZE = (15, 10)
    FIG_FORMAT = 'png'
    SAVE_FLAG = True
    SHOW_FLAG = False
    MOUSE_HOOVER = False
    MATPLOTLIB_BACKEND = "Agg" # , "Qt4Agg"


class CalculusConfig(object):
    SYMBOLIC_CALCULATIONS_FLAG = False

    # Normalization configuration
    WEIGHTS_NORM_PERCENT = 95

    # Options: "auto_eigenvals",  "auto_disease", "auto_epileptogenicity", "auto_excitability",
    # or "user_defined", in which case we expect a number equal to from 1 to hypothesis.n_regions
    LSA_METHOD = "1D" # other options: "2D", "auto"
    EIGENVECTORS_NUMBER_SELECTION = "auto_eigenvals"
    WEIGHTED_EIGENVECTOR_SUM = True
    INTERACTIVE_ELBOW_POINT = False

    MIN_SINGLE_VALUE = np.finfo("single").min
    MAX_SINGLE_VALUE = np.finfo("single").max
    MAX_INT_VALUE = np.iinfo(np.int64).max
    MIN_INT_VALUE = np.iinfo(np.int64).max


class SimulatorConfig(object):
    USE_TIME_DELAYS_FLAG = True
    MODE = GenericConfig.MODE_TVB


class HypothesisConfig(object):

    def __init__(self, head_folder=None):
        self.head_folder = head_folder


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
