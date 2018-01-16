import importlib
from abc import ABCMeta
import numpy as np
from tvb_epilepsy.base.constants.module_constants import MAX_SINGLE_VALUE, MIN_SINGLE_VALUE
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, make_float
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution import \
    ProbabilityDistribution
from tvb_epilepsy.service.probability_distribution_factory import compute_pdf_params
from tvb_epilepsy.service.stochastic_parameter_factory import get_val_key_for_first_keymatch_in_dict


class StochasticParameterBase(Parameter, ProbabilityDistribution):
    __metaclass__ = ABCMeta

    def __init__(self, name="Parameter", low=MIN_SINGLE_VALUE, high=MAX_SINGLE_VALUE, loc=0.0, scale=1.0, p_shape=()):
        Parameter.__init__(self, name, low, high, p_shape)
        self.loc = loc
        self.scale = scale

    def __repr__(self):
        d = {"01. name": self.name,
             "02. low": self.low,
             "03. high": self.high,
             }
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    #TODO: this should be cleaned
    def _string_generator(self):
        exec_str = "from tvb_epilepsy.base.model.statistical_models.stochastic_parameter " + \
                   "import generate_stochastic_parameter"
        eval_str = "generate_stochastic_parameter(" + self.name + \
                   ", probability_distribution=" + self.type + \
                   ", optimize=False)"
        d = {"exec": exec_str, "eval": eval_str}
        return d

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

    def _update_params(self, use="scipy", **params):
        self.loc = make_float(params.pop("loc", self.loc))
        self.scale = make_float(params.pop("scale", self.scale))
        self.update_params(self.loc, self.scale, use=use, **params)

    def _update_loc_scale(self, use="scipy", **target_stats):
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
            self._update_params(use=use)


def generate_stochastic_parameter(name="Parameter", low=-MAX_SINGLE_VALUE, high=MAX_SINGLE_VALUE, loc=0.0, scale=1.0,
                                  p_shape=(), probability_distribution="uniform", optimize_pdf=False, use="scipy",
                                  **pdf_params):
    pdf_module = importlib.import_module("tvb_epilepsy.base.model.statistical_models.probability_distributions." +
                                         probability_distribution.lower() + "_distribution")
    thisProbabilityDistribution = eval("pdf_module." + probability_distribution.title() + "Distribution")
    if optimize_pdf:
        pdf_params = compute_pdf_params(probability_distribution.lower(), pdf_params, loc, scale, use)

    class StochasticParameter(StochasticParameterBase, thisProbabilityDistribution):
        def __init__(self, name="Parameter", low=-MAX_SINGLE_VALUE, high=MAX_SINGLE_VALUE, loc=0.0, scale=1.0,
                     p_shape=(), use="scipy", **pdf_params):
            StochasticParameterBase.__init__(self, name, low, high, loc, scale, p_shape)
            thisProbabilityDistribution.__init__(self, **pdf_params)
            self._update_params(use=use)

        def __str__(self):
            return StochasticParameterBase.__str__(self) + "\n" \
                   + "\n".join(thisProbabilityDistribution.__str__(self).splitlines()[1:])

        def _scipy(self):
            return self.scipy(self.loc, self.scale)

        def _numpy(self, size=(1,)):
            return self.numpy(self.loc, self.scale, size)

    return StochasticParameter(name, low, high, loc, scale, p_shape, **pdf_params)


if __name__ == "__main__":
    sp = generate_stochastic_parameter("test", probability_distribution="gamma", optimize=False, shape=1.0, scale=2.0)
    print(sp)
