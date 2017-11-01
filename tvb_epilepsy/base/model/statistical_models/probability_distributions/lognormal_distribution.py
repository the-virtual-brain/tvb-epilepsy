
import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class LognormalDistribution(ContinuousProbabilityDistribution):

    def __init__(self, mu=0.0, sigma=1.0):
        self.name = "lognormal"
        self.scipy_name = "lognorm"
        self.params = {"mu": np.float(mu), "sigma": np.float(sigma)}
        self.n_params = len(self.params)
        self.constraint_string = "sigma > 0"
        if not(self.constraint()):
            raise_value_error("Constraint for " + self.name + " distribution " + self.constraint_string +
                              "\nwith parameters " + str(self.params) + " is not satisfied!")
        self.mu = self.calc_mu()
        self.median = self.calc_median()
        self.mode = self.calc_mode()
        self.var = self.calc_var()
        self.std = self.calc_std()
        self.skew = self.calc_skew()
        self.exkurt = self.calc_exkurt()

    def constraint(self):
        return self.params["sigma"] > 0.0

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(self.params["sigma"], loc=loc, scale=np.exp(self.params["mu"]))

    def calc_mu(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="m")
        else:
            return np.exp(self.params["mu"] + self.params["sigma"] ** 2 / 2.0)

    def calc_median(self, use="scipy"):
        if "":
            return self.scipy().median()
        else:
            return np.exp(self.params["mu"])

    def calc_mode(self):
        return np.exp(self.params["mu"] - self.params["sigma"] ** 2)

    def calc_var(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().var()
        else:
            sigma2 = self.params["sigma"] ** 2
            return (np.exp(sigma2) - 1.0) * np.exp(2.0 * self.params["mu"] + self.params["sigma"] ** 2)

    def calc_std(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().std()
        else:
            return np.sqrt(self.calc_var(self, use=use))

    def calc_skew(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="s")
        else:
            sigma2exp = np.exp(self.params["sigma"] ** 2)
            return (sigma2exp + 2.0) * np.sqrt(sigma2exp - 1.0)

    def calc_exkurt(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="k")
        else:
            sigma2 = self.params["sigma"] ** 2
            return np.exp(4.0 * sigma2) + 2.0 * np.exp(3.0 * sigma2) + 3.0 * np.exp(2.0 * sigma2) - 6.0
