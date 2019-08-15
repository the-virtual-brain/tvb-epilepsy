from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from tvb_fit.base.model.probabilistic_models.parameters.base import get_x_arg_for_param_distrib

from tvb_utils.data_structures_utils import formal_repr
from tvb_utils.log_error_utils import raise_not_implemented_error

TransformedProbabilisticParameterBaseAttributes = ["name", "type", "low", "high", "mean", "median", "mode",
                                                "var", "std", "skew", "kurt", "star"]
TransformedProbabilisticParameterBaseStarAttributes = ["star_low", "star_high", "star_mean", "star_median", "star_mode",
                                                    "star_var", "star_std", "star_skew", "star_kurt"]


class TransformedProbabilisticParameterBase(object):
    __metaclass__ = ABCMeta

    name = ""
    type = ""
    star = None

    def __init__(self, name, type, star_parameter):
        self.name = name.split("_star")[0]
        self.type = type
        self.star = star_parameter
        self.star.name = self.star.name.split("_star")[0] + "_star"

    def __getattr__(self, attr):
        if attr in TransformedProbabilisticParameterBaseAttributes:
            return super(TransformedProbabilisticParameterBase, self).__getattr__(attr)
        elif attr.find("star_") == 0:
            return getattr(self.star, attr.split("star_")[1])
        else:
            return getattr(self.star, attr)

    def __setattr__(self, attr, value):
        if attr in ["name", "type", "star"]:
            super(TransformedProbabilisticParameterBase, self).__setattr__(attr, value)
            return self
        else:
            setattr(self.star, attr, value)
            return self

    def _repr(self,  d=OrderedDict()):
        for ikey, key in enumerate(TransformedProbabilisticParameterBaseAttributes[:-1]):
            d.update({key: getattr(self, key)})
        for ikey, key in enumerate(TransformedProbabilisticParameterBaseStarAttributes):
            d.update({key: getattr(self, key)})
        d.update({"star parameter": str(self.star)})
        return d

    def __repr__(self, d=OrderedDict()):
        return formal_repr(self, self._repr())

    def __str__(self):
        return self.__repr__()

    # Overwrite any of the methods below for any specific implementation...
    @property
    def low(self):
        return self.star.low

    @abstractmethod
    def high(self):
        return self.star.high

    @property
    def mean(self):
        return self.star.mean

    @property
    def median(self):
        return self.star.median

    @property
    def mode(self):
        return self.star.mode

    @property
    def var(self):
        return self.star.var

    @property
    def std(self):
        return self.star.std

    @property
    def skew(self):
        return self.star.skew

    @property
    def kurt(self):
        return self.star.kurt

    def numpy(self, size=()):
        return self.star.numpy(size)

    def scipy_method(self, method, loc=0.0, scale=1.0, *args, **kwargs):
        return self.star.scipy_method(method, loc, scale, *args, **kwargs)


class NegativeLognormal(TransformedProbabilisticParameterBase, object):

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

    def _repr(self, d = OrderedDict()):
        d.update({"0. max": str(self.max)})
        d.update(super(NegativeLognormal, self)._repr(d))
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
    def skew(self):
        return -self.star.skew

    def scipy_method(self, method, *args, **kwargs):
        if method in ["rvs", "ppf", "isf", "stats", "moment", "median", "mean", "interval"]:
            return self.max - self.star.scipy_method(method, *args, **kwargs)
        elif method in ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"]:
            x, args, kwargs = get_x_arg_for_param_distrib(self, *args, **kwargs)
            args[0] = self.max - x
            pdf = self.star.scipy_method(method,  *args, **kwargs)[1]
            return x, pdf
        else:
            raise_not_implemented_error("Scipy method " + method +
                                        " is not implemented for transformed parameter " + self.name + "!")

    def numpy(self, size=()):
        return self.max - self._numpy(self.loc, self.scale, size)