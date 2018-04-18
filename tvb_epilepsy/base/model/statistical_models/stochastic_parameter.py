from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np
from tvb_epilepsy.base.constants.config import CalculusConfig
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, make_float, linspace_broadcast, \
    get_val_key_for_first_keymatch_in_dict
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.computations.probability_distributions.probability_distribution import ProbabilityDistribution


class StochasticParameterBase(Parameter, ProbabilityDistribution):
    __metaclass__ = ABCMeta

    def __init__(self, name="Parameter", low=CalculusConfig.MIN_SINGLE_VALUE, high=CalculusConfig.MAX_SINGLE_VALUE,
                 loc=0.0, scale=1.0, p_shape=()):
        Parameter.__init__(self, name, low, high, p_shape)
        self.loc = loc
        self.scale = scale

    def __str__(self):
        return Parameter.__str__(self) + "\n" \
               + "\n".join(ProbabilityDistribution.__str__(self).splitlines()[1:])

    def calc_mean(self, use="scipy"):
        return self._calc_mean(self.loc, self.scale, use)

    def calc_median(self, use="scipy"):
        return self._calc_median(self.loc, self.scale, use)

    def calc_mode(self):
        return self._calc_mode(self.loc, self.scale)

    def calc_std(self, use="scipy"):
        return self._calc_std(self.loc, self.scale, use)

    def calc_var(self, use="scipy"):
        return self._calc_var(self.loc, self.scale, use)

    def calc_skew(self, use="scipy"):
        return self._calc_skew(self.loc, self.scale, use)

    def calc_kurt(self, use="scipy"):
        return self._calc_kurt(self.loc, self.scale, use)

    @property
    def scipy(self):
        return self._scipy(self.loc, self.scale)

    def scipy_method(self, method, *args, **kwargs):
        if method in ["rvs", "ppf", "isf", "stats", "moment", "median", "mean", "interval"]:
            return self._scipy_method(method, self.loc, self.scale, *args, **kwargs)
        elif method in ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"]:
            x = kwargs.get("x", None)
            if x is None and len(args) > 0:
                # Assume that the first argument is x
                x = args[0]
            else:
                # Generate our own x
                x = np.arange(self.low, self.high, 100.0 / (self.high - self.low))
            args = tuple([x] + list(args[1:]))
            return x, self._scipy_method(method, self.loc, self.scale, *args, **kwargs)
        else:
            raise_not_implemented_error("Scipy method " + method +
                                        " is not implemented for parameter " + self.name + "!")

    def numpy(self, size=()):
        return self._numpy(self.loc, self.scale, size)

    def _update_params(self, use="scipy", **params):
        self.loc = make_float(params.pop("loc", self.loc))
        self.scale = make_float(params.pop("scale", self.scale))
        self.update_params(self.loc, self.scale, use=use, **params)
        return self

    def _confirm_support(self):
        p_star = (self.low - self.loc) / self.scale
        p_star_cdf = self.scipy.cdf(p_star)
        if np.any(p_star_cdf + np.finfo(np.float).eps <= 0.0):  #
            raise_value_error("Lower limit of " + self.name + " base distribution outside support!: " +
                              "\n(self.low-self.loc)/self.scale) = " + str(p_star) +
                              "\ncdf(self.low-self.loc)/self.scale) = " + str(p_star_cdf))
        p_star = (self.high - self.loc) / self.scale
        p_star_cdf = self.scipy.cdf(p_star)
        if np.any(p_star_cdf - np.finfo(np.float).eps) >= 1.0:
            self.logger.warning("Upper limit of base " + self.name + "  distribution outside support!: " +
                                "\n(self.high-self.loc)/self.scale) = " + str(p_star) +
                                "\ncdf(self.high-self.loc)/self.scale) = " + str(p_star_cdf))

    def update_loc_scale(self, use="scipy", **target_stats):
        param_m = self._calc_mean(use=use)
        target_m = self._calc_mean(use=use)
        param_s = self._calc_std(use=use)
        target_s = self._calc_std(use=use)
        if len(target_stats) > 0:
            m_fun = lambda scale: self._calc_mean(scale=scale, use=use)
            m, pkey = get_val_key_for_first_keymatch_in_dict(self.name,
                                                             ["def", "median", "med", "mode", "mod", "mean", "mu", "m"],
                                                             **target_stats)
            if m is not None:
                target_m = m
                if pkey in ["median", "med"]:
                    m_fun = lambda scale: self._calc_median(scale=scale, use=use)
                elif pkey in ["mode", "mod"]:
                    m_fun = lambda scale: self._calc_mode(scale=scale)
            s, pkey = get_val_key_for_first_keymatch_in_dict(self.name, ["var", "v", "std", "sig", "sigma", "s"],
                                                             **target_stats)
            if s is not None:
                target_s = s
                if pkey in ["var", "v"]:
                    target_s = np.sqrt(target_s)
        if np.any(param_m != target_m) or np.any(param_s != target_s):
            self.scale = target_s / param_s
            temp_m = m_fun(scale=self.scale)
            self.loc = target_m - temp_m
            self._confirm_support()
            self._update_params(use=use)
        return self

# TODO: this should move to examples
# if __name__ == "__main__":
#     sp = generate_stochastic_parameter("test", probability_distribution="gamma", optimize=False, shape=1.0, scale=2.0)
#     initialize_logger(__name__).info(sp)


TransformedStochasticParameterBaseAttributes = ["name", "type", "low", "high", "mean", "median", "mode",
                                                "var", "std", "skew", "kurt", "star"]

TransformedStochasticParameterBaseStarAttributes = ["star_low", "star_high", "star_mean", "star_median", "star_mode",
                                                    "star_var", "star_std", "star_skew", "star_kurt"]


class TransformedStochasticParameterBase(object):
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
        if attr in TransformedStochasticParameterBaseAttributes:
            return super(TransformedStochasticParameterBase, self).__getattr__(attr)
        elif attr.find("star_") == 0:
            return getattr(self.star, attr.split("star_")[1])
        else:
            return getattr(self.star, attr)

    def __setattr__(self, attr, value):
        if attr in ["name", "type", "star"]:
            super(TransformedStochasticParameterBase, self).__setattr__(attr, value)
            return self
        else:
            setattr(self.star, attr, value)
            return self

    def _repr(self,  d=OrderedDict()):
        for ikey, key in enumerate(TransformedStochasticParameterBaseAttributes[:-1]):
            d.update({key: getattr(self, key)})
        for ikey, key in enumerate(TransformedStochasticParameterBaseStarAttributes):
            d.update({key: getattr(self, key)})
        d.update({"star parameter": str(self.star)})
        return d

    def __repr__(self, d=OrderedDict()):
        return formal_repr(self, self._repr())

    def __str__(self):
        return self.__repr__()

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
            x = kwargs.get("x", None)
            if x is None and len(args) > 0:
                # Assume that the first argument is x and needs transformation
                x = np.array(args[0])
                x_transf = self.max - x
            else:
                # Generate our own x
                x_transf = linspace_broadcast(
                                np.maximum(self.low, self.scipy_method("ppf", 0.01)),
                                np.minimum(self.high, self.scipy_method("ppf", 0.99)), 100)
                x = self.max - x_transf
            args = tuple([x_transf] + list(args[1:]))
            pdf = self.star.scipy_method(method,  *args, **kwargs)[0]
            return x, pdf
        else:
            raise_not_implemented_error("Scipy method " + method +
                                        " is not implemented for transformed parameter " + self.name + "!")

    def numpy(self, size=()):
        return self.max - self._numpy(self.loc, self.scale, size)
