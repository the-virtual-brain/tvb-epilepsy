import numpy as np

from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, X1_DEF, K_DEF
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.probability_distributions.uniform_distribution \
                                                                                              import UniformDistribution
from tvb_epilepsy.base.model.statistical_models.probability_distributions.normal_distribution import NormalDistribution
from tvb_epilepsy.base.model.statistical_models.probability_distributions.gamma_distribution import GammaDistribution



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
        self.parameters.append(parameters.get("x1eq", Parameter("x1eq",
                                                                low=parameters.get("x1eq_lo", X1_DEF),
                                                                high=parameters.get("x1eq_hi", X1_EQ_CR_DEF),
                                                                probability_distribution=parameters.get("x1eq_pdf",
                                                                  NormalDistribution(
                                                                    mu=parameters.get("x1eq", (X1_EQ_CR_DEF-X1_DEF)/2),
                                                                    sigma=parameters.get("x1eq_sig", 0.1))),
                                                                shape=(self.n_regions, ))))
        parameter = parameters.get("K")
        if parameter is None:
            probability_distribution = parameters.get("K_pdf")
            if probability_distribution is None:
                probability_distribution = GammaDistribution()
                probability_distribution.compute_and_update_params({"mode": parameters.get("K_def", K_DEF),
                                                                    "std": parameters.get("K_sig", K_DEF)})
            parameter = Parameter("K", low=parameters.get("K_lo", 0.01),
                                       high=parameters.get("K_hi", 2.0),
                                       probability_distribution=probability_distribution,
                                       shape=(self.n_regions,))
        self.parameters.append(parameter)
        # tau1_def = parameters.get("tau1_def", 0.5)
        low = parameters.get("tau1_lo", 0.1)
        high = parameters.get("tau1_hi", 0.9)
        self.parameters.append(parameters.get("tau1", Parameter("tau1",
                                                                low=low,
                                                                high=high,
                                                                probability_distribution=
                                                                  UniformDistribution(a=low, b=high),
                                                                shape=(1,))))
        parameter = parameters.get("tau0")
        if parameter is None:
            tau0_def = parameters.get("tau0_def", 30.0)
            probability_distribution = parameters.get("tau0_pdf")
            if probability_distribution is None:
                probability_distribution = GammaDistribution()
                probability_distribution.compute_and_update_params({"mode": tau0_def,
                                                                    "std": parameters.get("K_sig", tau0_def)})
                parameter = Parameter("tau0", low=parameters.get("tau0_lo", 3.0),
                                              high=parameters.get("tau0_hi", 30000.0),
                                              probability_distribution=probability_distribution,
                                              shape=(1,))
        self.parameters.append(parameter)
        # Coupling:
        parameter = parameters.get("EC")
        if parameter is None:
            probability_distribution = parameters.get("EC_pdf")
            if probability_distribution is None:
                structural_connectivity = parameters.get("structural_connectivity",
                                                         10 ** -3 * np.ones((n_regions, n_regions)))
                EC_sig = parameters.get("EC_sig", structural_connectivity.flatten().median())
                probability_distributions = []
                for sc in structural_connectivity.flatten().tolist():
                    probability_distribution = GammaDistribution()
                    probability_distribution.compute_and_update_params({"mode": sc, "std": EC_sig})
                    probability_distributions.append(probability_distribution)
                probability_distribution = np.reshape(probability_distributions, structural_connectivity.shape)
            parameter = Parameter("EC",
                                  low=parameters.get("ec_lo", 10 ** -6),
                                  high=parameters.get("ec_hi", 100.0),
                                  probability_distribution=probability_distribution,
                                  shape=(self.n_regions, self.n_regions))
        self.parameters.append(parameter)

        # Integration:
        parameter = parameters.get("sig_eq")
        if parameter is None:
            sig_eq_def = parameters.get("sig_eq_def", 0.1)
            probability_distribution = parameters.get("sig_eq_pdf")
            if probability_distribution is None:
                probability_distribution = GammaDistribution()
                probability_distribution.compute_and_update_params({"mode": sig_eq_def,
                                                                    "std": parameters.get("sig_eq_sig", sig_eq_def)})
                parameter = Parameter("sig_eq",
                                      low=parameters.get("sig_eq_lo", sig_eq_def / 10.0),
                                      high=parameters.get("sig_eq_hi", 3 * sig_eq_def),
                                      probability_distribution=probability_distribution,
                                      shape=(1,))
        self.parameters.append(parameter)

        # Observation model
        parameter = parameters.get("eps")
        if parameter is None:
            eps_def = parameters.get("eps_def", 0.1)
            probability_distribution = parameters.get("eps_pdf")
            if probability_distribution is None:
                probability_distribution = GammaDistribution()
                probability_distribution.compute_and_update_params({"mode": eps_def,
                                                                    "std": parameters.get("eps_sig", eps_def)})
                parameter = Parameter("eps",
                                      low=parameters.get("eps_lo", 0.0),
                                      high=parameters.get("eps_hi", 1.0),
                                      probability_distribution=probability_distribution,
                                      shape=(1,))
        self.parameters.append(parameter)

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
