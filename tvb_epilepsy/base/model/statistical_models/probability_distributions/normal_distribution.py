
from collections import OrderedDict

import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string, construct_import_path
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class NormalDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.type = "normal"
        self.scipy_name = "norm"
        self.numpy_name = "normal"
        self.mean = make_float(params.get("mean", params.get("loc", 0.0)))
        self.sigma = make_float(make_float(params.get("sigma", params.get("scale", 1.0))))
        self.constraint_string = "sigma > 0"
        self.__update_params__(mean=self.mean, sigma=self.sigma)
        self.context_str = "from " + construct_import_path(__file__) + " import NormalDistribution"
        self.create_str = "NormalDistribution('" + self.type + "')"
        self.update_str = "obj.update_params()"

    def pdf_params(self, parametrization="mean-sigma"):
        p = OrderedDict()
        if isequal_string(parametrization, "scipy") or isequal_string(parametrization, "numpy"):
            p.update(zip(["loc", "scale"], [self.mean, self.sigma]))
            return p
        else:
            p.update(zip(["mean", "sigma"], [self.mean, self.sigma]))
            return p

    def scale_params(self, loc=0.0, scale=1.0):
        return self.mean + loc, self.sigma * scale

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               mean=make_float(params.get("mean", params.get("loc", self.mean))),
                               sigma=make_float(params.get("sigma", params.get("scale", self.sigma))))

    def constraint(self):
        # By default expr >= 0
        return np.array(self.sigma).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(loc=self.mean+loc, scale=self.sigma*scale)

    def numpy(self, loc=0.0, scale=1.0, size=(1,)):
        return lambda: nr.normal(self.mean + loc, self.sigma * scale, size=size)

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        return self.mean + loc

    def calc_median_manual(self, loc=0.0, scale=1.0):
        return self.mean + loc

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        return self.mean + loc

    def calc_var_manual(self, loc=0.0, scale=1.0):
        return self.sigma**2 * scale**2

    def calc_std_manual(self, loc=0.0, scale=1.0):
        return self.sigma * scale

    def calc_skew_manual(self, loc=0.0, scale=1.0):
        return 0.0

    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        return 0.0
