
import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.discrete_probability_distribution  \
                                                                                  import DiscreteProbabilityDistribution


class BernoulliDistribution(DiscreteProbabilityDistribution):

    def __init__(self, p=0.5):
        self.name = "bernoulli"
        self.scipy_name = "bernoulli"
        self.params = {"p": np.float(p)}
        self.constraint_string = "0 < p < 1"
        self.__update_params__(**self.params)

    def update_params(self, **params):
        self.__update_params__(**self.params)

    def constraint(self):
        return self.params["p"] > 0.0 and self.params["p"] < 1.0

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(self.params["p"], loc=loc, scale=scale)

    def calc_mu_manual(self):
        return self.params["p"]

    def calc_median_manual(self):
        if self.params["p"] < 0.5:
            return 0.0
        elif self.params["p"] > 0.5:
            return 1.0
        else:
            return 0.5

    def calc_mode_manual(self):
        if self.params["p"] < 0.5:
            return 0.0
        elif self.params["p"] > 0.5:
            return 1.0
        else:
            warning("The mode of bernoulli distribution for p=0.5 consists of two values (0.0 and 1.0)!")
            return 0.0, 1.0

    def calc_var_manual(self):
        return self.params["p"] * (1 - self.params["p"])

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        return (1.0 - 2.0 * self.params["p"]) / self.calc_std_manual()

    def calc_exkurt_manual(self):
        var = self.calc_var_manual()
        return (1.0 - 6.0 * var) / var
