
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import make_float
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class ExponentialDistribution(ContinuousProbabilityDistribution):

    def __init__(self, scale=1.0):
        self.name = "exponential"
        self.scipy_name = "expon"
        self.numpy_name = "exponential"
        self.params = {"scale": make_float(scale)}
        self.constraint_string = "scale > 0"
        self.__update_params__(**self.params)
        self.lamda = 1.0/ scale
        self.rate = self.lamda

    def __str__(self):
        this_str = super(ExponentialDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. rate or lamda" + " = " + str(self.rate) + "}"
        return this_str

    def update_params(self, **params):
        self.__update_params__(**self.params)
        self.lamda = 1.0 / self.params["scale"]
        self.rate = self.lamda

    def constraint(self):
        return np.all(self.params["scale"] > 0)

    def scipy(self, loc=0.0, scale=1.0):
        return ss.expon(loc=loc, scale=self.params["scale"])

    def numpy(self, size=(1,)):
        return lambda: nr.exponential(self.params["scale"], size=size)

    def calc_mean_manual(self):
        return self.params["scale"]

    def calc_median_manual(self):
        warning("Approximate calculation for median of chisquare distribution!")
        return self.params["scale"] * np.log(2)

    def calc_mode_manual(self):
        return 0.0

    def calc_var_manual(self):
        return self.params["scale"] ** 2

    def calc_std_manual(self):
        return self.params["scale"]

    def calc_skew_manual(self):
        return 2.0

    def calc_kurt_manual(self):
        return 6.0
