
from collections import OrderedDict

import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class NormalDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.type = "normal"
        self.scipy_name = "norm"
        self.numpy_name = "normal"
        self.mean = make_float(params.get("mean", params.get("loc", 0.0)))
        self.sigma = make_float(make_float(params.get("sigma", params.get("scale", 1.0))))
        self.loc = self.mean
        self.scale = self.sigma
        self.constraint_string = "sigma > 0"
        self.__update_params__(mean=self.mean, sigma=self.sigma)

    def pdf_params(self, parametrization="mean-sigma"):
        p = OrderedDict()
        if isequal_string(parametrization, "scipy") or isequal_string(parametrization, "numpy"):
            p.update(zip(["loc", "scale"], [self.loc, self.scale]))
            return p
        else:
            p.update(zip(["mean", "sigma"], [self.mean, self.sigma]))
            return p

    def update_params(self, **params):
        self.__update_params__(mean=make_float(params.get("mean", params.get("loc", self.mean))),
                               sigma=make_float(params.get("sigma", params.get("scale", self.sigma))))

    def constraint(self):
        # By default expr >= 0
        return np.array(self.sigma).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(loc=self.mean, scale=self.sigma)

    def numpy(self, size=(1,)):
        return lambda: nr.normal(self.mean, self.sigma, size=size)

    def calc_mean_manual(self):
        return self.mean

    def calc_median_manual(self, ):
        return self.mean

    def calc_mode_manual(self):
        return self.mean

    def calc_var_manual(self):
        return self.sigma ** 2

    def calc_std_manual(self, ):
        return self.sigma

    def calc_skew_manual(self):
        return 0.0

    def calc_kurt_manual(self):
        return 0.0
