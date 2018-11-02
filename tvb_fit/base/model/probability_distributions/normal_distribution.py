from collections import OrderedDict
import numpy as np
import numpy.random as nr
import scipy.stats as ss
from tvb_fit.base.model.probability_distributions import ProbabilityDistributionTypes
from tvb_fit.base.utils.data_structures_utils import make_float, isequal_string
from tvb_fit.base.model.probability_distributions.continuous_probability_distribution \
    import ContinuousProbabilityDistribution


class NormalDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.type = ProbabilityDistributionTypes.NORMAL
        self.scipy_name = "norm"
        self.numpy_name = ProbabilityDistributionTypes.NORMAL
        self.mu = make_float(params.get("mu", params.get("loc", 0.0)))
        self.sigma = make_float(make_float(params.get("sigma", params.get("scale", 1.0))))
        self.constraint_string = "sigma > 0"
        self.__update_params__(mu=self.mu, sigma=self.sigma)

    def pdf_params(self, parametrization="mu-sigma"):
        p = OrderedDict()
        if isequal_string(parametrization, "scipy") or isequal_string(parametrization, "numpy"):
            p.update(zip(["loc", "scale"], [self.mu, self.sigma]))
            return p
        else:
            p.update(zip(["mu", "sigma"], [self.mu, self.sigma]))
            return p

    def scale_params(self, loc=0.0, scale=1.0):
        return self.mu + loc, self.sigma * scale

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               mu=make_float(params.get("mu", params.get("loc", self.mu))),
                               sigma=make_float(params.get("sigma", params.get("scale", self.sigma))))

    def constraint(self):
        # By default expr >= 0
        return np.array(self.sigma).flatten() - np.finfo(np.float64).eps

    def _scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(loc=self.mu+loc, scale=self.sigma*scale)

    def _numpy(self, loc=0.0, scale=1.0, size=(1,)):
        return lambda: nr.normal(self.mu + loc, self.sigma * scale, size=size)

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        return self.mu + loc

    def calc_median_manual(self, loc=0.0, scale=1.0):
        return self.mu + loc

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        return self.mu + loc

    def calc_var_manual(self, loc=0.0, scale=1.0):
        return (self.sigma * scale) ** 2

    def calc_std_manual(self, loc=0.0, scale=1.0):
        return self.sigma * scale

    def calc_skew_manual(self, loc=0.0, scale=1.0):
        return 0.0

    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        return 0.0
