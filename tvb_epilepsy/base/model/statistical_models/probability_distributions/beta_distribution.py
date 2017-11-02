
import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class BetaDistribution(ContinuousProbabilityDistribution):

    def __init__(self, alpha=1.0, beta=1.0):
        self.name = "beta"
        self.scipy_name = "beta"
        self.params = {"alpha": np.float(alpha), "beta": np.float(beta)}
        self.constraint_string = "alpha > 0 and beta > 0"
        self.__update_params__(**self.params)

    def update_params(self, **params):
        self.__update_params__(**self.params)

    def constraint(self):
        return self.params["alpha"] > 0.0 and self.params["beta"] > 0.0

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(self.params["alpha"], self.params["beta"], loc=loc, scale=scale)

    def calc_mu(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="m")
        else:
            return self.params["alpha"] / (self.params["alpha"] + self.params["beta"])

    def calc_median(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().median()
        else:
            if self.params["alpha"] > 1.0 and self.params["beta"] > 1.0:
                warning("Approximate calculation for median of beta distribution!")
                return (self.params["alpha"] - 1.0/3) / (self.params["alpha"] + self.params["beta"] - 2.0/3)
            else:
                raise_value_error("No closed form of median for beta distribution for alpha or beta <= 1.0!")

    def calc_mode(self):
        if self.params["alpha"] > 1.0 and self.params["beta"] > 1.0:
            return (self.params["alpha"] - 1.0) / (self.params["alpha"] + self.params["beta"] - 2.0)
        else:
            raise_value_error("No closed form of mode for beta distribution for alpha or beta <= 1.0!")

    def calc_var(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().var()
        else:
            a_plus_b = self.params["alpha"] + self.params["beta"]
            return self.params["alpha"] * self.params["beta"] / a_plus_b ** 2 / (a_plus_b + 1.0)

    def calc_std(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().std()
        else:
            return np.sqrt(self.calc_var(use=use))

    def calc_skew(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="s")
        else:
            a_plus_b = self.params["alpha"] + self.params["beta"]
            return 2.0 * (self.params["beta"] - self.params["alpha"]) * np.sqrt(a_plus_b + 1.0) / \
                   (a_plus_b + 2) / np.sqrt(self.params["alpha"] * self.params["beta"])

    def calc_exkurt(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="k")
        else:
            a_plus_b = self.params["alpha"] + self.params["beta"]
            ab = self.params["alpha"] * self.params["beta"]
            return 6.0 * (((self.params["alpha"] - self.params["beta"]) ** 2) * (a_plus_b + 1.0) -
                          ab * (a_plus_b + 2.0)) / ab / (a_plus_b + 2.0) / (a_plus_b + 3.0)
