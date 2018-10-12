import numpy as np
from tvb_fit.base.config import CalculusConfig
from tvb_fit.base.utils.log_error_utils import raise_value_error
from tvb_fit.base.utils.data_structures_utils import extract_dict_stringkeys, \
    get_val_key_for_first_keymatch_in_dict
from tvb_fit.base.model.probability_distributions import ProbabilityDistributionTypes
from tvb_fit.base.model.probabilistic_models.parameters.base import ProbabilisticParameterBase
from tvb_fit.base.model.probabilistic_models.parameters.transformed_parameters import NegativeLognormal
from tvb_fit.service.probability_distribution_factory import compute_pdf_params, probability_distribution_factory


#TODO: This could be turned into a builder once it is stable


# This function takes position or keyword arguments of the form "param" or "name_param" and sets the default parameters
# of a stochastic parameter in the form "name_param" if name_flag = True, or "param" otherwise,
# ready to enter to the stochastic parameter generation function
# The argument pdf_params targets the distribution "side"of a stochastic parameter instance, whereas,
# the rest of the parameters target the loc and scale of the stochastic parameter.
# The values for std, lo and hi can be callables of mean.

def set_parameter_defaults(name, _pdf="normal", _shape=(), _lo=CalculusConfig.MIN_SINGLE_VALUE,
                           _hi=CalculusConfig.MAX_SINGLE_VALUE, _mean=None,
                           _std=None, pdf_params={}, remove_name=False, **kwargs):
    if remove_name:
        out_name = lambda pkey: pkey
    else:
        out_name = lambda pkey: "_".join([name, pkey])
    defaults = {}
    defaults.update({out_name("pdf"): kwargs.pop("_".join([name, "pdf"]), kwargs.pop("pdf", _pdf))})
    defaults.update({out_name("shape"): kwargs.pop("_".join([name, "shape"]), kwargs.pop("shape", _shape))})
    defaults.update({out_name("pdf_params"): pdf_params})
    if _mean is None:
        _mean, pkey = \
            get_val_key_for_first_keymatch_in_dict(name, ["def", "median", "med", "mode", "mod", "mean", "mu", "m"],
                                                   **kwargs)
        if _mean is not None:
            if pkey in ["def", "mu", "m", "mean"]:
                defaults.update({out_name("mean"): _mean})
            elif pkey in ["median", "med"]:
                defaults.update({out_name("median"): _mean})
            elif pkey in ["mode", "mod"]:
                defaults.update({out_name("mode"): _mean})
    else:
        defaults.update({out_name("mean"): _mean})
    pkey = "std"
    if _std is None:
        _std, pkey = get_val_key_for_first_keymatch_in_dict(name, ["var", "v", "std", "sig", "sigma", "s"], **kwargs)
    if _std is not None:
        if pkey in ["var", "v"]:
            pkey = "var"
        elif pkey in ["std", "sig", "sigma", "s"]:
            pkey = "std"
        if callable(_std) and _mean is not None:  # std can be a function of mean
            _std = np.abs(_std(_mean))
        defaults.update({out_name(pkey): _std})
    for this_pval, pkey, pkeys in zip([_lo, _hi],
                                      ["lo", "hi"],
                                      [["lo", "low", "min"], ["hi", "high", "max"]]):
        pval = get_val_key_for_first_keymatch_in_dict(name, pkeys, **kwargs)[0]
        if pval is None:
            pval = this_pval
        if callable(pval) and _mean is not None:
            pval = pval(_mean)
        defaults.update({out_name(pkey): pval})
    return defaults


def set_parameter(name, use="scipy", **kwargs):
    parameter = kwargs.pop(name, None)
    # assign the mean value if parameter is numeric
    if isinstance(parameter, (int, long, float)) or (isinstance(parameter, np.ndarray)
                                                     and np.issubdtype(np.dtype, np.number)):
        kwargs.update({"_".join([name, "def"]): parameter})
    defaults = {}
    # Get all keyword arguments that correspond to that parameter name
    defaults.update(extract_dict_stringkeys(kwargs, name + "_"))
    # Generate defaults and eventually the parameter:
    defaults = set_parameter_defaults(name, pdf_params=defaults.pop("_".join([name, "pdf_params"]), {}),
                                      remove_name=True, **defaults)
    # If there is a dictionary of pdf parameters, there has to be optimization of the pdf shape as well
    pdf_params = defaults.pop("pdf_params", {})
    if len(pdf_params) > 0:
        optimize_pdf = True
    else:
        optimize_pdf = False
    # Generate the parameter with or without optimization of its shape:
    parameter = generate_probabilistic_parameter(name, probability_distribution=defaults.pop("pdf"),
                                                 p_shape=defaults.pop("shape"),
                                                 low=defaults.pop("lo"),
                                                 high=defaults.pop("hi"),
                                                 optimize_pdf=optimize_pdf, use=use, **pdf_params)
    # Update parameter's loc and scale if necessary by moving and/or scaling it accordingly
    if len(defaults) > 0:
        parameter.update_loc_scale(use=use, **defaults)
    return parameter


def generate_probabilistic_parameter(name="Parameter", low=-CalculusConfig.MAX_SINGLE_VALUE,
                                     high=CalculusConfig.MAX_SINGLE_VALUE, loc=0.0, scale=1.0,
                                     p_shape=(), probability_distribution=ProbabilityDistributionTypes.UNIFORM,
                                     optimize_pdf=False, use="scipy", **target_params):
    thisProbabilityDistribution = probability_distribution_factory(probability_distribution.lower(), get_instance=False)

    class ProbabilisticParameter(ProbabilisticParameterBase, thisProbabilityDistribution):
        def __init__(self, name="Parameter", low=-CalculusConfig.MAX_SINGLE_VALUE, high=CalculusConfig.MAX_SINGLE_VALUE,
                     loc=0.0, scale=1.0, p_shape=(), use="scipy", **target_params):
            ProbabilisticParameterBase.__init__(self, name, low, high, loc, scale, p_shape)
            thisProbabilityDistribution.__init__(self, **target_params)
            success = True
            for p_key, p_val in target_params.items():
                if np.any(p_val != getattr(self, p_key)):
                    success = False
            if success is False:
                if optimize_pdf:
                    pdf_params = compute_pdf_params(probability_distribution.lower(), target_params, loc, scale, use)
                    thisProbabilityDistribution.__init__(self, **pdf_params)
                    success = True
                    for p_key, p_val in target_params.items():
                        if np.any(np.abs(p_val - getattr(self, p_key)) > 0.1):
                            success = False
            if success is False:
                raise_value_error("Cannot generate probability distribution of type " + probability_distribution +
                                  " with parameters " + str(target_params) + " !")
                self._update_params(use=use)

    return ProbabilisticParameter(name, low, high, loc, scale, p_shape, use, **target_params)


def generate_normal_parameter(name, mean, low, high, sigma=None, sigma_scale=2, p_shape=(), use="scipy"):
    if sigma is None:
        sigma = np.minimum(np.abs(mean - low), np.abs(high - mean)) / sigma_scale
    return generate_probabilistic_parameter(name, low, high, loc=0.0, scale=1.0, p_shape=p_shape,
                                            probability_distribution=ProbabilityDistributionTypes.NORMAL,
                                            optimize_pdf=False, use=use, **{"mu": mean, "sigma": sigma})


def generate_lognormal_parameter(name, mean, low, high, sigma=None, sigma_scale=2, p_shape=(), use="scipy"):
    if sigma is None:
        sigma = np.abs(mean - low) / sigma_scale
    logsm21 = np.log((sigma / mean) ** 2 + 1)
    mu = np.log(mean) - 0.5 * logsm21
    sigma = np.sqrt(logsm21)
    return generate_probabilistic_parameter(name, low, high, loc=0.0, scale=1.0, p_shape=p_shape,
                                            probability_distribution=ProbabilityDistributionTypes.LOGNORMAL,
                                            optimize_pdf=False, use=use, **{"mu": mu, "sigma": sigma})


def generate_negative_lognormal_parameter(name, mean, low, high, sigma=None, sigma_scale=2, p_shape=(), use="scipy"):
    parameter = generate_lognormal_parameter(name.split("_star")[0]+"_star", high - mean, 0.0, high - low,
                                             sigma, sigma_scale, p_shape, use)

    return NegativeLognormal(parameter.name.split("_star")[0], "NegativeLognormal", parameter, high)
