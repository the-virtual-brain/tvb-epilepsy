
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
        self.n_params = len(self.params)
        self.constraint_string = "int(k) > 0"
        if not(self.constraint()):
            raise_value_error("Constraint for " + self.name + " distribution " + self.constraint_string +
                              "\nwith parameters " + str(self.params) + " is not satisfied!")
        self.df = self.params["k"]
        self.mu = self.calc_mu()
        self.median = self.calc_median()
        self.mode = self.calc_mode()
        self.var = self.calc_var()
        self.std = self.calc_std()
        self.skew = self.calc_skew()
        self.exkurt = self.calc_exkurt()

    def __str__(self):
        this_str = super(ChisquareDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. degrees of freedom" + " = " + str(self.df) + "}"
        return this_str

    def constraint(self):
        return self.params["k"] > 0

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(self.params["k"], loc=loc, scale=scale)

    def calc_mu(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="m")
        else:
            return self.params["k"]

    def calc_median(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().median()
        else:
            warning("Approximate calculation for median of chisquare distribution!")
            return self.params["k"] * (1 - 2.0 / (9 * self.params["k"])) ** 3

    def calc_mode(self):
        return np.max([self.params["k"] - 2, 0])

    def calc_var(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().var()
        else:
            return 2 * self.params["k"]

    def calc_std(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().std()
        else:
            return np.sqrt(self.calc_var(use=use))

    def calc_skew(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="s")
        else:
            return np.sqrt(8.0 / self.params["k"])

    def calc_exkurt(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="k")
        else:
            return 12.0 / self.params["k"]
