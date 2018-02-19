import numpy as np
from tvb_epilepsy.base.constants.config import CalculusConfig
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import StochasticParameterBase
from tvb_epilepsy.base.utils.data_structures_utils import extract_dict_stringkeys, \
    get_val_key_for_first_keymatch_in_dict
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.service.probability_distribution_factory import compute_pdf_params, probability_distribution_factory
from tvb_epilepsy.base.computations.probability_distributions import ProbabilityDistributionTypes

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


def set_parameter(name, use="manual", **kwargs):
    parameter = kwargs.pop(name, None)
    # load parameter if it is a file
    if not (isinstance(parameter, Parameter)):
        defaults = {}
        # Get all keyword arguments that correspond to that parameter name
        defaults.update(extract_dict_stringkeys(kwargs, name + "_"))
        # assign the mean value if parameter is numeric
        if isinstance(parameter, (int, long, float)) or (isinstance(parameter, np.ndarray)
                                                         and np.issubdtype(np.dtype, np.number)):
            kwargs.update({"_".join([name, "def"]): parameter})
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
        parameter = generate_stochastic_parameter(name, probability_distribution=defaults.pop("pdf"),
                                                  p_shape=defaults.pop("shape"),
                                                  low=defaults.pop("lo"),
                                                  high=defaults.pop("hi"),
                                                  optimize_pdf=optimize_pdf, use=use, **pdf_params)
        # Update parameter's loc and scale if necessary by moving and/or scaling it accordingly
        if len(defaults) > 0:
            parameter._update_loc_scale(use=use, **defaults)
    return parameter


def generate_stochastic_parameter(name="Parameter", low=-CalculusConfig.MAX_SINGLE_VALUE, 
                                  high=CalculusConfig.MAX_SINGLE_VALUE, loc=0.0, scale=1.0,
                                  p_shape=(), probability_distribution=ProbabilityDistributionTypes.UNIFORM,
                                  optimize_pdf=False, use="scipy", **target_params):
    thisProbabilityDistribution = probability_distribution_factory(probability_distribution.lower(), get_instance=False)

    class StochasticParameter(StochasticParameterBase, thisProbabilityDistribution):
        def __init__(self, name="Parameter", low=-CalculusConfig.MAX_SINGLE_VALUE, high=CalculusConfig.MAX_SINGLE_VALUE,
                     loc=0.0, scale=1.0, p_shape=(), use="scipy", **target_params):
            StochasticParameterBase.__init__(self, name, low, high, loc, scale, p_shape)
            thisProbabilityDistribution.__init__(self, **target_params)
            success = True
            for p_key, p_val in target_params.iteritems():
                if np.any(p_val != getattr(self, p_key)):
                    success = False
            if success is False:
                if optimize_pdf:
                    pdf_params = compute_pdf_params(probability_distribution.lower(), target_params, loc, scale, use)
                    thisProbabilityDistribution.__init__(self, **pdf_params)
                    success = True
                    for p_key, p_val in target_params.iteritems():
                        if np.any(np.abs(p_val - getattr(self, p_key)) > 0.1):
                            success = False
            if success is False:
                raise_value_error("Cannot generate probability distribution of type " + probability_distribution +
                                  " with parameters " + str(target_params) + " !")
                self._update_params(use=use)

        def __str__(self):
            return StochasticParameterBase.__str__(self) + "\n" \
                   + "\n".join(thisProbabilityDistribution.__str__(self).splitlines()[1:])

        def _scipy(self):
            return self.scipy(self.loc, self.scale)

        def _numpy(self, size=(1,)):
            return self.numpy(self.loc, self.scale, size)

    return StochasticParameter(name, low, high, loc, scale, p_shape, **target_params)
