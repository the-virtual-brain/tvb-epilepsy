
import sys

import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


MAX_VALUE = sys.float_info.max
MIN_VALUE = sys.float_info.min


class UniformDistribution(ContinuousProbabilityDistribution):

    def __init__(self, a=MIN_VALUE, b=MAX_VALUE):
        self.name = "uniform"
        self.scipy_name = "uniform"
        self.params = {"a": np.float(a), "b": np.float(b)}
        self.n_params = len(self.params)
        self.constraint_string = "a < b"
        if not(self.constraint()):
            raise_value_error("Constraint for " + self.name + " distribution " + self.constraint_string +
                              "\nwith parameters " + str(self.params) + " is not satisfied!")
        self.mu = self.calc_mu()
        self.median = self.calc_median()
        self.mode = None
        self.var = self.calc_var()
        self.std = self.calc_std()
        self.skew = self.calc_skew()
        self.exkurt = self.calc_exkurt()

    def constraint(self):
        return self.params["a"] < self.params["b"]

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(loc=self.params["a"], scale=self.params["b"]-self.params["a"])

    def calc_mu(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="m")
        else:
            return 0.5 * (self.params["a"] + self.params["b"])

    def calc_median(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().median()
        else:
            return 0.5 * (self.params["a"] + self.params["b"])

    def calc_mode(self):
        warning("Uniform distribution does not have a definite mode!")

    def calc_var(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().var()
        else:
            return ((self.params["b"] - self.params["a"]) ** 2) / 12.0

    def calc_std(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().std()
        else:
            return (self.params["b"] - self.params["a"]) / np.sqrt(12)

    def calc_skew(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="s")
        else:
            return 0.0

    def calc_exkurt(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="k")
        else:
            return -1.2
