
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import make_float
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class GammaDistribution(ContinuousProbabilityDistribution):

    def __init__(self, shape=1.0, scale=1.0):
        self.name = "gamma"
        self.scipy_name = "gamma"
        self.numpy_name = "gamma"
        self.params = {"shape": make_float(shape), "scale": make_float(scale)}
        self.constraint_string = "shape > 0 and scale > 0"
        self.__update_params__(**self.params)
        self.k = shape
        self.theta = scale
        self.alpha = shape
        self.beta = 1.0 / scale

    def __str__(self):
        this_str = super(GammaDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. k" + " = " + str(self.k) + \
                    "\n" + "14. theta" + " = " + str(self.theta) + \
                    "\n" + "15. alpha" + " = " + str(self.alpha) + \
                    "\n" + "16. beta" + " = " + str(self.beta) + "}"
        return this_str

    def update_params(self, **params):
        self.__update_params__(**self.params)
        self.k = self.params["shape"]
        self.theta = self.params["scale"]
        self.alpha = self.params["shape"]
        self.beta = 1.0 / self.params["scale"]

    def constraint(self):
        return np.all(self.params["shape"] > 0.0) and np.all(self.params["scale"] > 0.0)

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(self.params["shape"], loc=loc, scale=self.params["scale"])

    def numpy(self, size=(1,)):
        return lambda: nr.gamma(self.params["shape"], self.params["scale"], size=size)

    def calc_mean_manual(self):
        return self.params["shape"] * self.params["scale"]

    def calc_median_manual(self):
        warning("Gamma distribution does not have a simple closed form median! Returning nan!")
        return np.nan

    def calc_mode_manual(self):
        if self.params["shape"] >= 1.0:
            return (self.params["shape"] - 1.0) * self.params["scale"]
        else:
            warning("Mode cannot be calculate for gamma distribution when the shape parameter is smaller than 1.0! "
                    "Returning nan!")
            return np.nan

    def calc_var_manual(self):
        return self.params["shape"] * self.params["scale"] ** 2

    def calc_std_manual(self):
       return np.sqrt(self.params["shape"]) * self.params["scale"]

    def calc_skew_manual(self):
        return 2.0 / np.sqrt(self.params["shape"])

    def calc_exkurt_manual(self):
        return 6.0 / self.params["shape"]
