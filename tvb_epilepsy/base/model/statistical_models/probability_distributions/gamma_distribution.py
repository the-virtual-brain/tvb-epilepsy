
import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class GammaDistribution(ContinuousProbabilityDistribution):

    def __init__(self, shape=1.0, scale=1.0):
        self.name = "gamma"
        self.scipy_name = "gamma"
        self.params = {"shape": np.float(shape), "scale": np.float(scale)}
        self.n_params = len(self.params)
        self.params_alpha_beta = {"k": shape, "theta": 1.0}
        self.params_alpha_beta = {"alpha": shape, "beta": 1.0 / scale}
        self.constraint_string = "shape > 0 and scale > 0"
        if not(self.constraint()):
            raise_value_error("Constraint for " + self.name + " distribution " + self.constraint_string +
                              "\nwith parameters " + str(self.params) + " is not satisfied!")
        self.mu = self.calc_mu()
        self.median = None
        self.mode = self.calc_mode()
        self.var = self.calc_var()
        self.std = self.calc_std()
        self.skew = self.calc_skew()
        self.exkurt = self.calc_exkurt()

    def __str__(self):
        this_str = super(GammaDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. alpha-beta parameters" + " = " + str(self.params_alpha_beta) + \
                    "\n" + "14. k-theta parameters" + " = " + str(self.params_k_theta) + "}"
        return this_str

    def constraint(self):
        return self.params["shape"] > 0.0 and self.params["scale"] > 0.0

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(self.params["shape"], loc=loc, scale=self.params["scale"])

    def calc_mu(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="m")
        else:
            return self.params["shape"] * self.params["scale"]

    def calc_median(self, use="scipy"):
        warning("Gamma distribution does not have a simple closed form median!")

    def calc_mode(self):
        if self.params["shape"] >= 1.0:
            return (self.params["shape"] - 1.0) * self.params["scale"]
        else:
            warning("Mode cannot be calculate for gamma distribution when the shape parameter is smaller than 1.0!")

    def calc_var(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().var()
        else:
            return self.params["shape"] * self.params["scale"] ** 2

    def calc_std(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().std()
        else:
            return np.sqrt(self.params["shape"]) * self.params["scale"]

    def calc_skew(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="s")
        else:
            return 2.0 / np.sqrt(self.params["shape"])

    def calc_exkurt(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="k")
        else:
            return 6.0 / self.params["shape"]
