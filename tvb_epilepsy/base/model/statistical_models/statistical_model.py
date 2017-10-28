import numpy as np

from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, X1_DEF, K_DEF
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.model.statistical_models.parameter import Parameter
from tvb_epilepsy.base.h5_model import convert_to_h5_model


class StatisticalModel(object):

    def __init__(self, name, parameters, n_regions=0):

        self.n_regions = n_regions

        if isinstance(name, basestring):
            self.name = name
        else:
            raise_value_error("Statistical model's name " + str(name) + " is not a string!")

        # Parameter setting:
        self.parameters = []
        # Generative model:
        # Epileptor:
        self.parameters.append(Parameter("x1eq",
                                    low=parameters.get("x1eq_lo", X1_DEF),
                                    high=parameters.get("x1eq_hi", X1_EQ_CR_DEF),
                                    loc=parameters.get("x1eq_loc"),
                                    scale=parameters.get("x1eq_sc"),
                                    shape=(self.n_regions, ),
                                    pdf="normal"))
        self.parameters.append(Parameter("K",
                                    low=parameters.get("K_lo", 0.01),
                                    high=parameters.get("K_hi", 2.0),
                                    loc=parameters.get("K_loc", K_DEF),
                                    scale=parameters.get("K_sc", K_DEF),
                                    shape=(1,),
                                    pdf="gamma"))
        tau1_def = parameters.get("tau1_def", 0.5)
        self.parameters.append(Parameter("tau1",
                                    low=parameters.get("tau1_lo", 0.1),
                                    high=parameters.get("tau1_hi", 0.9),
                                    loc=parameters.get("tau1_loc", tau1_def),
                                    scale=parameters.get("tau1_sc", tau1_def),
                                    shape=(1,),
                                    pdf="uniform"))
        tau0_def = parameters.get("tau0_def", 30.0)
        self.parameters.append(Parameter("tau0",
                                    low=parameters.get("tau0_lo", 3.0),
                                    high=parameters.get("tau0_hi", 30000.0),
                                    loc=parameters.get("tau0_loc", tau0_def),
                                    scale=parameters.get("tau0_sc", tau0_def),
                                    shape=(1,),
                                    pdf="gamma"))
        # Coupling:
        structural_connectivity = parameters.get("structural_connectivity", 10 ** -6 * np.ones((n_regions, n_regions)))
        self.parameters.append(Parameter("EC",
                                    low=parameters.get("ec_lo", 10 ** -6),
                                    high=parameters.get("ec_hi", 100.0),
                                    loc=structural_connectivity,
                                    scale=structural_connectivity,
                                    shape=(self.n_regions, self.n_regions),
                                    pdf="gamma"))
        # Integration:
        sig_eq_def = parameters.get("sig_eq_def", 0.1)
        self.parameters.append(Parameter("sig_eq",
                                    low=parameters.get("sig_eq_lo", sig_eq_def / 10.0),
                                    high=parameters.get("sig_eq_hi", 3 * sig_eq_def),
                                    loc=parameters.get("sig_eq_loc", sig_eq_def),
                                    scale=parameters.get("sig_eq_sc", sig_eq_def),
                                    shape=(1,),
                                    pdf="gamma"))

        # Observation model
        self.parameters.append(Parameter("eps",
                                    low=parameters.get("eps_lo", 0.0),
                                    high=parameters.get("eps_hi", 1.0),
                                    loc=parameters.get("eps_loc", 0.1),
                                    scale=parameters.get("eps_sc", 0.1),
                                    shape=(1,),
                                    pdf="gamma"))

        self.n_parameters = len(self.parameters)

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "StatisicalModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)
