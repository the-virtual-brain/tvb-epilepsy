import os

import numpy as np

from tvb_epilepsy.base.constants.module_constants import MAX_SINGLE_VALUE, MIN_SINGLE_VALUE
from tvb_epilepsy.base.h5_model import read_h5_model
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import generate_stochastic_parameter
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import extract_dict_stringkeys, isequal_string


def set_parameter(name, optimize=True, **kwargs):
    parameter = kwargs.get(name, None)
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
        # Get all keyword arguments that correspond to that parameter
        defaults.update(extract_dict_stringkeys(kwargs, name, remove=False)[0])
        # Add other keyword arguments that do NOT correspond to OTHER parameters, e.g., "mean", etc
        defaults.update(extract_dict_stringkeys(kwargs, "_", remove=True)[0])
        # assign the mean value if parameter is numeric
        if isinstance(parameter, (int, long, float)) or (isinstance(parameter, np.ndarray)
                                                         and np.issubdtype(np.dtype, np.number)):
            kwargs.update({"_".join([name, "def"]): parameter})
        # Generate defaults and eventually the parameter:
        defaults = set_parameter_defaults(name, **defaults)
        parameter = generate_stochastic_parameter(name,
                                                  probability_distribution=defaults.pop("_".join([name, "pdf"])),
                                                  p_shape=defaults.pop("_".join([name, "shape"])),
                                                  low=defaults.pop("_".join([name, "lo"])),
                                                  high=defaults.pop("_".join([name, "hi"])),
                                                  optimize=optimize, **defaults)
    return parameter


def set_parameter_defaults(name, pdf="normal", shape=(), lo=MIN_SINGLE_VALUE, hi=MAX_SINGLE_VALUE, mean=None, std=None,
                           **kwargs):
    defaults = dict()
    defaults.update({name + "_pdf": kwargs.pop("_".join([name, "pdf"]), kwargs.pop("pdf", pdf))})
    defaults.update({name + "_shape": kwargs.pop("_".join([name, "shape"]), kwargs.pop("shape", shape))})
    for pstring in ["median", "mode", "def", "mu", "m", "mean"]:
        p_value = kwargs.pop("_".join([name, pstring]), kwargs.pop(pstring, None))
        if p_value is not None:
            mean = p_value
            if isequal_string(pstring, "def") or isequal_string(pstring, "mu") or isequal_string(pstring, "m"):
                pstring = "mean"
            break
    if mean is not None:
        defaults.update({"_".join([name, pstring]): mean})
    for pstring in ["sig", "sigma", "s", "var", "std"]:
        std = kwargs.pop("_".join([name, pstring]), kwargs.pop(pstring, std))
        if std is not None:
            if pstring != "var":
                pstring = "std"
            break
    if std is None:
        if mean is not None:
            std = np.abs(mean / 3.0)
    else:
        if callable(std) and mean is not None:
            std = np.abs(std(mean))
        if pstring == "var":
            std = np.sqrt(np.abs(std))
    if std is not None and not(np.any(std == 0.0)):
        defaults.update({"_".join([name, pstring]): std})

    def set_low_high(val, strings):
        value = val
        for pstring in strings:
            p_value = kwargs.pop("_".join([name, pstring]), kwargs.pop(pstring, None))
            if p_value is not None:
                value = p_value
                break
        if callable(value) and mean is not None:
            value = value(mean)
        return value

    defaults.update({"_".join([name, "lo"]): set_low_high(lo, ["lo", "low", "min"])})
    defaults.update({"_".join([name, "hi"]): set_low_high(lo, ["hi", "high", "max"])})
    return defaults