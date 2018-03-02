import numpy as np
import scipy.stats as ss
import numpy.random as nr
from collections import OrderedDict
from tvb_epilepsy.base.computations.probability_distributions import ProbabilityDistributionTypes
from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string
from tvb_epilepsy.base.computations.probability_distributions.continuous_probability_distribution \
    import ContinuousProbabilityDistribution

DEFAULT_LOW_VALUE = 0.0
DEFAULT_HIGH_VALUE = 1.0


class UniformDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.type = ProbabilityDistributionTypes.UNIFORM
        self.scipy_name = ProbabilityDistributionTypes.UNIFORM
        self.numpy_name = ProbabilityDistributionTypes.UNIFORM
        self.constraint_string = "a < b"
        self.a = make_float(params.get("a", params.get("low", params.get("loc", DEFAULT_LOW_VALUE))))
        self.b = make_float(params.get("b", params.get("high",
                                                       params.get("scale",
                                                                  2.0 * DEFAULT_HIGH_VALUE) - DEFAULT_HIGH_VALUE)))
        self.__update_params__(a=self.a, b=self.b)
        self.low = self.a
        self.high = self.b

    def pdf_params(self, parametrization="a-b"):
        p = OrderedDict()
        if isequal_string(parametrization, "scipy"):
            p.update(zip(["loc", "scale"], [self.a, self.b - self.a]))
            return p
        elif isequal_string(parametrization, "numpy"):
            p.update(zip(["low", "high"], [self.low, self.high]))
            return p
        else:
            p.update(zip(["a", "b"], [self.a, self.b]))
            return p

    def scale_params(self, loc=0.0, scale=1.0):
        return self.a + loc, self.a + loc + (self.b - self.a) * scale

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               a=make_float(params.get("a", params.get("low", params.get("loc", self.a)))),
                               b=make_float(params.get("b", params.get("high",
                                                                       params.get("scale", self.b - self.a) + self.a))))
        self.low = self.a
        self.high = self.b

    def constraint(self):
        # By default expr >= 0
        return np.array(self.b).flatten() - np.array(self.a).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        a, b = self.scale_params(loc, scale)
        return getattr(ss, self.scipy_name)(loc=a, scale=b - a)

    def numpy(self, loc=0.0, scale=1.0, size=(1,)):
        a, b = self.scale_params(loc, scale)
        return lambda: nr.uniform(a, b, size=size)

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        a, b = self.scale_params(loc, scale)
        return 0.5 * (a + b)

    def calc_median_manual(self, loc=0.0, scale=1.0):
        a, b = self.scale_params(loc, scale)
        return 0.5 * (a + b)

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        # TODO: find a way to mute this warning...
        # self.logger.warning("Uniform distribution does not have a definite mode! Returning nan!")
        return np.nan

    def calc_var_manual(self, loc=0.0, scale=1.0):
        a, b = self.scale_params(loc, scale)
        return ((b - a) ** 2) / 12.0

    def calc_std_manual(self, loc=0.0, scale=1.0):
        a, b = self.scale_params(loc, scale)
        return (b - a) / np.sqrt(12)

    def calc_skew_manual(self, loc=0.0, scale=1.0):
        return 0.0

    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        return -1.2
