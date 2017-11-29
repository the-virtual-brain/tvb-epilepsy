
import os

import numpy as np

from tvb_epilepsy.base.constants.module_constants import MAX_SINGLE_VALUE, MIN_SINGLE_VALUE
from tvb_epilepsy.base.h5_model import read_h5_model
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import generate_stochastic_parameter
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import extract_dict_stringkeys, isequal_string


def set_parameter(name, optimize=True, **kwargs):
    parameter = kwargs.pop(name, None)
    # load parameter it is a file
    if isinstance(parameter, basestring):
        if os.path.isfile(parameter):
            try:
                parameter = read_h5_model(parameter).convert_from_h5_model()
            except:
                warning("Failed to read parameter " + name + " from file path " + parameter + "!\n" +
                        "Proceeding with generating it!")
    if not(isinstance(parameter, Parameter)):
        defaults = {}
        # Get all keyword arguments that correspond to that parameter name
        defaults.update(extract_dict_stringkeys(kwargs, name + "_"))
        # assign the mean value if parameter is numeric
        if isinstance(parameter, (int, long, float)) or (isinstance(parameter, np.ndarray)
                                                         and np.issubdtype(np.dtype, np.number)):
            kwargs.update({"_".join([name, "def"]): parameter})
        # Generate defaults and eventually the parameter:
        defaults = set_parameter_defaults(name, defaults.pop("_".join([name, "pdf_params"]), {}), name_flag=False,
                                          **defaults)
        # Strip the parameter name from the dictionary keys:
        parameter = generate_stochastic_parameter(name,
                                                      probability_distribution=defaults.pop("pdf"),
                                                      p_shape=defaults.pop("shape"),
                                                      low=defaults.pop("lo"),
                                                      high=defaults.pop("hi"),
                                                      optimize=optimize, **defaults)
    return parameter


# This function takes position or keyword arguments of the form "param" or "name_param" and sets the default parameters
# of a stochastic parameter in the form "name_param" if name_flag = True, or "param" otherwise,
# ready to enter to the stochastic parameter generation function
# The argument pdf_params has priority over mean, std or other variations thereof. It can be used to set other
# possible parameters such as scale, shape, kurt etc
# The values for std, lo and hi can be callables of mean.
def set_parameter_defaults(name, _pdf="normal", _shape=(), _lo=MIN_SINGLE_VALUE, _hi=MAX_SINGLE_VALUE, _mean=None,
                           _std=None, pdf_params={}, name_flag=True, **kwargs):
    if name_flag:
        out_name = lambda pkey: "_".join([name, pkey])
    else:
        out_name = lambda pkey: pkey
    defaults = {}
    defaults.update({out_name("pdf"): kwargs.pop("_".join([name, "pdf"]), kwargs.pop("pdf", _pdf))})
    defaults.update({out_name("shape"): kwargs.pop("_".join([name, "shape"]), kwargs.pop("shape", _shape))})
    # A set of pdf_params has priority over mean and str:
    if len(pdf_params) > 0:
        for pkey, pval in pdf_params.iteritems():
            defaults.update({out_name(pkey.split(name + "_", 0)[-1]): pval})
    else:
        pkey="mean"
        if _mean is None:
            # go along mean and std and their kind...
            for pkey in ["def", "median", "mode", "mu", "m", "mean"]:
                _mean = kwargs.pop("_".join([name, pkey]), kwargs.pop(pkey, _mean))
                if _mean is not None:
                    if not(isequal_string(pkey, "median")) and not(isequal_string(pkey, "mode")):
                        pkey = "mean"
                    break
        if _mean is not None:
            defaults.update({out_name("mean"): _mean})
        pkey = "std"
        if _std is None:
            for pkey in ["sig", "sigma", "s", "var", "std"]:
                _std = kwargs.pop("_".join([name, pkey]), kwargs.pop(pkey, _std))
                if _std is not None:
                    if not(isequal_string(pkey, "sigma")) and not(isequal_string(pkey, "var")):
                        pkey = "std"
                    break
        if _std is not None:
            if callable(_std) and _mean is not None: # std can be a function of mean
                _std = np.abs(_std(_mean))
            if pkey == "var":
                _std = np.sqrt(np.abs(_std))
                pkey = "std"
            if not(np.any(_std == 0.0)):
                defaults.update({out_name(pkey): _std})

    def set_low_high(val, strings):
        value = val
        for pkey in strings:
            pval = kwargs.pop("_".join([name, pkey]), kwargs.pop(pkey, None))
            if pval is not None:
                value = pval
                break
        if callable(value) and _mean is not None:     # low and high can be functions of mean
            value = value(_mean)
        return value

    defaults.update({out_name("lo"): set_low_high(_lo, ["lo", "low", "min"])})
    defaults.update({out_name("hi"): set_low_high(_hi, ["hi", "high", "max"])})
    return defaults