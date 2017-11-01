
import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class ExponentialDistribution(ContinuousProbabilityDistribution):

    def __init__(self, scale=1.0):
        self.name = "exponential"
        self.scipy_name = "expon"
        self.params = {"scale": np.float(scale)}
        self.n_params = len(self.params)
        self.constraint_string = "scale > 0"
        if not(self.constraint()):
            raise_value_error("Constraint for " + self.name + " distribution " + self.constraint_string +
                              "\nwith parameters " + str(self.params) + " is not satisfied!")
        self.rate = 1.0 / self.params["scale"]
        self.lamda = self.rate
        self.mu = self.calc_mu()
        self.median = self.calc_median()
        self.mode = self.calc_mode()
        self.var = self.calc_var()
        self.std = self.calc_std()
        self.skew = self.calc_skew()
        self.exkurt = self.calc_exkurt()

    def __str__(self):
        this_str = super(ExponentialDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. rate or lamda" + " = " + str(self.rate) + "}"
        return this_str

    def constraint(self):
        return np.all(self.params["scale"] > 0)

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(loc=loc, scale=self.params["scale"])

    def calc_mu(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="m")
        else:
            return self.params["scale"]

    def calc_median(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().median()
        else:
            warning("Approximate calculation for median of chisquare distribution!")
            return self.params["scale"] * np.log(2)

    def calc_mode(self):
        return 0.0

    def calc_var(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().var()
        else:
            return self.params["scale"] ** 2

    def calc_std(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().std()
        else:
            return self.params["scale"]

    def calc_skew(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="s")
        else:
            return 2.0

    def calc_exkurt(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="k")
        else:
            return 6.0
