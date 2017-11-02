
import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class ChisquareDistribution(ContinuousProbabilityDistribution):

    def __init__(self, k=1):
        self.name = "chisquare"
        self.scipy_name = "chi"
        self.params = {"k": np.int(k)}
        self.constraint_string = "int(k) > 0"
        self.__update_params__(**self.params)
        self.df = self.k

    def __str__(self):
        this_str = super(ChisquareDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. degrees of freedom" + " = " + str(self.df) + "}"
        return this_str

    def update_params(self, **params):
        self.__update_params__(**self.params)
        self.df = self.k

    def constraint(self):
        return self.params["k"] > 0

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(self.params["k"], loc=loc, scale=scale)

    def calc_mu_manual(self):
        return self.params["k"]

    def calc_median_manual(self):
        warning("Approximate calculation for median of chisquare distribution!")
        return self.params["k"] * (1 - 2.0 / (9 * self.params["k"])) ** 3

    def calc_mode_manual(self):
        return np.max([self.params["k"] - 2, 0])

    def calc_var_manual(self):
        return 2 * self.params["k"]

    def calc_std_manual(self):
        return np.sqrt(self.calc_va_manualr())

    def calc_skew_manual(self):
        return np.sqrt(8.0 / self.params["k"])

    def calc_exkurt_manual(self):
        return 12.0 / self.params["k"]
