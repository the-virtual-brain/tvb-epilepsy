
from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import raise_not_implemented_error, raise_value_error
from tvb_epilepsy.base.computations.probability_distributions import ProbabilityDistributionTypes
from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import TransformedStochasticParameterBase
from tvb_epilepsy.service.stochastic_parameter_builder import generate_stochastic_parameter


def generate_lognormal_parameter(name, mean, low, high, sigma=None, sigma_scale=3, p_shape=(), use="scipy"):
    if sigma is None:
        sigma = np.minimum(np.abs(high - mean), np.abs(mean - low)) / sigma_scale
    return generate_stochastic_parameter(name, low, high, loc=0.0, scale=1.0, p_shape=p_shape,
                                         probability_distribution=ProbabilityDistributionTypes.LOGNORMAL,
                                         optimize_pdf=True, use=use, **{"mean": mean/sigma, "skew": 0.0}). \
                                         update_loc_scale(use=use, **{"mean": mean, "std": sigma})


def generate_negative_lognormal_parameter(name, mean, low, high, sigma=None, sigma_scale=3, p_shape=(), use="scipy"):
    parameter = generate_lognormal_parameter(name.split("_star")[0]+"_star", high - mean, 0.0, high - low,
                                             sigma, sigma_scale, p_shape, use)

    class NegativeLognormal(TransformedStochasticParameterBase, object):

        def __init__(self, name, type, parameter, max):
            super(NegativeLognormal, self).__init__(name, type, parameter)
            self.max = max

        def __getattr__(self, attr):
            if attr == "max":
                return object.__setattr__(self, "max")
            else:
                return super(NegativeLognormal, self).__getattr__(attr)

        def __setattr__(self, attr, value):
            if attr == "max":
                object.__setattr__(self, "max", value)
                return self
            else:
                super(NegativeLognormal, self).__setattr__(attr, value)
                return self

        def __repr__(self):
            d = OrderedDict({"0. max": str(self.max)})
            d.update(super(NegativeLognormal, self).__repr__(d))
            return d

        @property
        def low(self):
            return self.max - self.star.high

        @property
        def high(self):
            return self.max - self.star.low

        @property
        def mean(self):
            return self.max - self.star.mean

        @property
        def median(self):
            return self.max - self.star.median

        @property
        def mode(self):
            return self.max - self.star.mode

        @property
        def var(self):
            return self.star.var

        @property
        def std(self):
            return self.star.std

        @property
        def skew(self):
            return -self.star.skew

        @property
        def kurt(self):
            return self.star.kurt

        def _scipy_method(self, method, loc=0.0, scale=1.0, *args, **kwargs):
            if method in ["rvs", "ppf", "isf", "stats", "moment", "median", "mean", "interval"]:
                return self.max - self.star._scipy_method(method, loc, scale, *args, **kwargs)
            elif method in ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"]:
                x = kwargs.get("x", None)
                if x is None and len(args) > 0:
                    x = args[0]
                if x is not None:
                    # Assume that the first argument is x and transform it
                    args = tuple([self.max - np.array(x)] + list(args[1:]))
                    return self.star._scipy_method(method, loc, scale, *args, **kwargs)
                else:
                    raise_value_error("Scipy method " + method + " for transformed parameter " + self.name +
                                      " cannot be executed due to missing argument x!")
            else:
                raise_not_implemented_error("Scipy method " + method +
                                            " is not implemented for transformed parameter " + self.name + "!")

        def numpy(self):
            return self.max - self._numpy(self.loc, self.scale)

    return NegativeLognormal(parameter.name.split("_star")[0], "NegativeLognormal", parameter, high)
