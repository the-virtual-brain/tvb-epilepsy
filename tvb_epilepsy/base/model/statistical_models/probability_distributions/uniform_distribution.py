
from collections import OrderedDict

import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.constants import MAX_SINGLE_VALUE
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution

DEFAULT_LOW_VALUE = -MAX_SINGLE_VALUE
DEFAULT_HIGH_VALUE = MAX_SINGLE_VALUE

class UniformDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.name = "uniform"
        self.scipy_name = "uniform"
        self.numpy_name = "uniform"
        self.constraint_string = "a < b"
        self.a = make_float(params.get("a", params.get("low", params.get("loc", DEFAULT_LOW_VALUE))))
        self.b = make_float(params.get("b", params.get("high",
                                                     params.get("scale", 2.0*DEFAULT_HIGH_VALUE) - DEFAULT_HIGH_VALUE)))
        self.__update_params__(a=self.a, b=self.b)
        self.low = self.a
        self.high = self.b
        self.loc = self.a
        self.scale = self.b - self.a

    def params(self, parametrization="a-b"):
        p = OrderedDict()
        if isequal_string(parametrization, "scipy"):
            p.update(zip(["loc", "scale"], [self.loc, self.scale]))
            return p
        elif isequal_string(parametrization, "numpy"):
            p.update(zip(["low", "high"], [self.low, self.high]))
            return p
        else:
            p.update(zip(["a", "b"], [self.a, self.b]))
            return p

    def update_params(self, **params):
        self.__update_params__(a=make_float(params.get("a", params.get("low", params.get("loc", self.a)))),
                               b=make_float(params.get("b", params.get("high",
                                                                       params.get("scale", self.scale) + self.a))))
        self.low = self.a
        self.high = self.b
        self.loc = self.a
        self.scale = self.b - self.a

    def constraint(self):
        # By default expr >= 0
        return np.array(self.b).flatten() - np.array(self.a).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(loc=self.a, scale=self.b-self.a)

    def numpy(self, size=(1,)):
        return lambda: nr.beta(self.a, self.b, size=size)

    def calc_mean_manual(self):
        return 0.5 * (self.a + self.b)

    def calc_median_manual(self):
        return 0.5 * (self.a + self.b)

    def calc_mode_manual(self):
        warning("Uniform distribution does not have a definite mode! Returning nan!")
        return np.nan

    def calc_var_manual(self):
        return ((self.b - self.a) ** 2) / 12.0

    def calc_std_manual(self):
        return (self.b - self.a) / np.sqrt(12)

    def calc_skew_manual(self):
        return 0.0

    def calc_kurt_manual(self):
        return -1.2
