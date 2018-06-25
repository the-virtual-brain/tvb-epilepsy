from tvb_infer.base.config import GenericConfig, InputConfig, OutputConfig, FiguresConfig
from tvb_infer.base.config import CalculusConfig as CalculusConfigBase


class CalculusConfig(CalculusConfigBase):

    # Options: "auto_eigenvals",  "auto_disease", "auto_epileptogenicity", "auto_excitability",
    # or "user_defined", in which case we expect a number equal to from 1 to hypothesis.n_regions
    LSA_METHOD = "1D" # other options: "2D", "auto"
    EIGENVECTORS_NUMBER_SELECTION = "auto_eigenvals"
    WEIGHTED_EIGENVECTOR_SUM = True


class Config(object):
    generic = GenericConfig()
    figures = FiguresConfig()
    calcul = CalculusConfig()

    def __init__(self, head_folder=None, data_mode=GenericConfig.MODE_JAVA,
                 raw_data_folder=None,
                 output_base=None, separate_by_run=False):
        self.input = InputConfig(head_folder, data_mode, raw_data_folder)
        self.out = OutputConfig(output_base, separate_by_run)
