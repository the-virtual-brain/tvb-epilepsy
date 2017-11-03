
import sys

import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import make_float
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


MAX_VALUE = sys.float_info.max
MIN_VALUE = sys.float_info.min


class UniformDistribution(ContinuousProbabilityDistribution):

    def __init__(self, a=MIN_VALUE, b=MAX_VALUE):
        self.name = "uniform"
        self.scipy_name = "uniform"
        self.params = {"a": make_float(a), "b": make_float(b)}
        self.constraint_string = "a < b"
        self.__update_params__(**self.params)

    def update_params(self, **params):
        self.__update_params__(**self.params)

    def constraint(self):
        return np.all(self.params["a"] < self.params["b"])

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(loc=self.params["a"], scale=self.params["b"]-self.params["a"])

    def calc_mu_manual(self):
        return 0.5 * (self.params["a"] + self.params["b"])

    def calc_median_manual(self):
        return 0.5 * (self.params["a"] + self.params["b"])

    def calc_mode_manual(self):
        warning("Uniform distribution does not have a definite mode! Returning nan!")
        return np.nan

    def calc_var_manual(self):
        return ((self.params["b"] - self.params["a"]) ** 2) / 12.0

    def calc_std_manual(self):
        return (self.params["b"] - self.params["a"]) / np.sqrt(12)

    def calc_skew_manual(self):
        return 0.0

    def calc_exkurt_manual(self):
        return -1.2
