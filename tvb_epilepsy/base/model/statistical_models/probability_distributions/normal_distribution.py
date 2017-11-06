
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.data_structures_utils import make_float
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class NormalDistribution(ContinuousProbabilityDistribution):

    def __init__(self, mu=0.0, sigma=1.0):
        self.name = "normal"
        self.scipy_name = "norm"
        self.numpy_name = "normal"
        self.params = {"mean": make_float(mu), "sigma": make_float(sigma)}
        self.constraint_string = "sigma > 0"
        self.__update_params__(**self.params)

    def update_params(self, **params):
        self.__update_params__(**self.params)

    def constraint(self):
        return np.all(self.params["sigma"] > 0.0)

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(loc=self.params["mean"], scale=self.params["sigma"])

    def numpy(self, size=(1,)):
        return lambda: nr.gamma(self.params["mean"], self.params["sigma"], size=size)

    def calc_mean_manual(self):
        return self.params["mean"]

    def calc_median_manual(self, ):
        return self.params["mean"]

    def calc_mode_manual(self):
        return self.params["mean"]

    def calc_var_manual(self):
        return self.params["std"] ** 2

    def calc_std_manual(self, ):
        return self.params["std"]

    def calc_skew_manual(self):
        return 0.0

    def calc_exkurt_manual(self):
        return 0.0
