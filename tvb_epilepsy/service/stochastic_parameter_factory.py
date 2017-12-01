
import os

import numpy as np

from tvb_epilepsy.base.constants.module_constants import MAX_SINGLE_VALUE, MIN_SINGLE_VALUE
from tvb_epilepsy.base.h5_model import read_h5_model
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import extract_dict_stringkeys, isequal_string


def get_val_key_for_firt_keymatch_in_dict(name, pkeys, **kwargs):
    pkeys += ["_".join([name, pkey]) for pkey in pkeys]
    temp = extract_dict_stringkeys(kwargs, pkeys, modefun="equal", break_after=1)
    if len(temp) > 0:
        return temp.values()[0], temp.keys()[0].split("_")[-1]
    else:
        return None, None


# This function takes position or keyword arguments of the form "param" or "name_param" and sets the default parameters
# of a stochastic parameter in the form "name_param" if name_flag = True, or "param" otherwise,
# ready to enter to the stochastic parameter generation function
# The argument pdf_params targets the distribution "side"of a stochastic parameter instance, whereas,
# the rest of the parameters target the loc and scale of the stochastic parameter.
# The values for std, lo and hi can be callables of mean.
def set_parameter_defaults(name, _pdf="normal", _shape=(), _lo=MIN_SINGLE_VALUE, _hi=MAX_SINGLE_VALUE, _mean=None,
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
        _mean, pkey = get_val_key_for_firt_keymatch_in_dict(name, ["def", "median", "med", "mode", "mod", "mean", "mu", "m"],
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
        _std, pkey = get_val_key_for_firt_keymatch_in_dict(name, ["var", "v", "std", "sig", "sigma", "s"], **kwargs)
    if _std is not None:
        if pkey in ["var", "v"]:
            pkey = "var"
        elif pkey in ["std", "sig", "sigma", "s"]:
            pkey = "std"
        if callable(_std) and _mean is not None: # std can be a function of mean
            _std = np.abs(_std(_mean))
        defaults.update({out_name(pkey): _std})
    for this_pval, pkey, pkeys in zip([_lo, _hi],
                                      ["lo", "hi"],
                                      [["lo", "low", "min"], ["hi", "high", "max"]]):
        pval = get_val_key_for_firt_keymatch_in_dict(name, pkeys, **kwargs)[0]
        if pval is None:
            pval = this_pval
        if callable(pval) and _mean is not None:
            pval = pval(_mean)
        defaults.update({out_name(pkey): pval})
    return defaults


def set_parameter(name, optimize_pdf=False, use="manual", **kwargs):
    parameter = kwargs.pop(name, None)
    # load parameter if it is a file
    if isinstance(parameter, basestring):
        if os.path.isfile(parameter):
            try:
                parameter = read_h5_model(parameter).convert_from_h5_model()
            except:
                warning("Failed to read parameter " + name + " from file path " + parameter + "!\n" +
                        "Proceeding with generating it!")
    if not(isinstance(parameter, Parameter)):
        from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import generate_stochastic_parameter
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
        # Strip the parameter name from the dictionary keys:
        parameter = generate_stochastic_parameter(name, probability_distribution=defaults.pop("pdf"),
                                                        p_shape=defaults.pop("shape"),
                                                        low=defaults.pop("lo"),
                                                        high=defaults.pop("hi"),
                                                        optimize_pdf=optimize_pdf, use=use,
                                                        **(defaults.pop("pdf_params", {})))
        if len(defaults) > 0:
            parameter._update_loc_scale(use=use, **defaults)
    return parameter