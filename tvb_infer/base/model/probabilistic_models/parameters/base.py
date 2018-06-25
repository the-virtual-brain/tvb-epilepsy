from abc import ABCMeta

import numpy as np

from tvb_infer.base.config import CalculusConfig
from tvb_infer.base.model.parameter import Parameter
from tvb_infer.base.model.probability_distributions.probability_distribution import ProbabilityDistribution
from tvb_infer.base.utils.data_structures_utils import make_float, get_val_key_for_first_keymatch_in_dict, \
    linspace_broadcast
from tvb_infer.base.utils.log_error_utils import raise_not_implemented_error, raise_value_error


class ProbabilisticParameterBase(Parameter, ProbabilityDistribution):
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
    def mean(self):
        return self.calc_mean()

    @property
    def median(self):
        return self.calc_median()

    @property
    def mode(self):
        return self.calc_mode()

    @property
    def var(self):
        return self.calc_var()

    @property
    def std(self):
        return self.calc_std()

    @property
    def skew(self):
        return self.calc_skew()

    @property
    def kurt(self):
        return self.calc_kurt()

    @property
    def scipy(self):
        return self._scipy(self.loc, self.scale)

    def scipy_method(self, method, *args, **kwargs):
        if method in ["rvs", "ppf", "isf", "stats", "moment", "median", "mean", "interval"]:
            return self._scipy_method(method, self.loc, self.scale, *args, **kwargs)
        elif method in ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"]:
            x, args, kwargs = get_x_arg_for_param_distrib(self, *args, **kwargs)
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


def get_x_arg_for_param_distrib(parameter, *args, **kwargs):
    args = list(args)
    x = kwargs.pop("x", None)
    if x is None or len(x) == 0:
        if len(args) > 0:
            x = np.array(args.pop(0))
        if x is None or len(x) == 0:
            # Generate our own x
            x = linspace_broadcast(np.maximum(parameter.low, parameter.scipy_method("ppf", 0.01)),
                                   np.minimum(parameter.high, parameter.scipy_method("ppf", 0.99)), 100)
    args = [x] + args
    return x, args, kwargs