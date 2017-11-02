
import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class NormalDistribution(ContinuousProbabilityDistribution):

    def __init__(self, mu=0.0, sigma=1.0):
        self.name = "normal"
        self.scipy_name = "norm"
        self.params = {"mu": np.float(mu), "sigma": np.float(sigma)}
        self.constraint_string = "sigma > 0"
        self.__update_params__(**self.params)

    def update_params(self, **params):
        self.__update_params__(**self.params)

    def constraint(self):
        return self.params["sigma"] > 0.0

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(loc=self.params["mu"], scale=self.params["sigma"])

    def calc_mu_manual(self):
        return self.params["mu"]

    def calc_median_manual(self, ):
        return self.params["mu"]

    def calc_mode_manual(self):
        return self.params["mu"]

    def calc_var_manual(self):
        return self.params["std"] ** 2

    def calc_std_manual(self, ):
        return self.params["std"]

    def calc_skew_manual(self):
        return 0.0

    def calc_exkurt_manual(self):
        return 0.0
