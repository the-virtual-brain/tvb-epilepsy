from collections import OrderedDict

import numpy as np

from tvb_fit.base.constants import PriorsModes, Target_Data_Type
from tvb_fit.base.utils.data_structures_utils import formal_repr
from tvb_fit.base.utils.log_error_utils import warning
from tvb_fit.base.model.probabilistic_models.parameters.base import ProbabilisticParameterBase
from tvb_fit.base.model.probabilistic_models.parameters.transformed_parameters import \
    TransformedProbabilisticParameterBase


class ProbabilisticModelBase(object):

    model_config = None
    name = "probabilistic_model"
    target_data_type = Target_Data_Type.EMPIRICAL.value
    priors_mode = PriorsModes.NONINFORMATIVE.value
    parameters = {}
    ground_truth = {}  # a dictionary of parameters'names (keys) to true values or functions to read them from the model_config

    @property
    def number_of_regions(self):
        if self.model_config is not None:
            return self.model_config.number_of_regions
        else:
            return 1

    @property
    def number_of_parameters(self):
        return len(self.parameters)

    def __init__(self, model_configuration, name='tvb',
                 target_data_type=Target_Data_Type.EMPIRICAL.value, priors_mode=PriorsModes.NONINFORMATIVE.value,
                 parameters={}, ground_truth={}):
        self.model_config = model_configuration
        self.target_data_type = target_data_type
        self.priors_mode = priors_mode
        self.parameters = parameters
        self.ground_truth = ground_truth

    def _repr(self, d=OrderedDict()):
        for ikey, (key, val) in enumerate(self.__dict__.items()):
            d.update({key:  val})
        return d

    def __repr__(self, d=OrderedDict()):
        return formal_repr(self, self._repr(d))

    def __str__(self):
        return self.__repr__()

    @property
    def number_of_total_parameters(self):
        nparams = 0
        for p in self.parameters.values():
            nparams += np.maximum(1, np.prod(p.p_shape))
        return nparams

    def get_parameter(self, parameter_name):
        parameter = self.parameters.get(parameter_name, None)
        if parameter is None:
            warning("Ground truth value for parameter " + parameter_name + " was not found!")
        return parameter

    def get_truth(self, parameter_name):
        truth = self.ground_truth.get(parameter_name, np.nan)
        if truth is np.nan and self.target_data_type == Target_Data_Type.SYNTHETIC.value:
            truth = getattr(self.model_config, parameter_name, np.nan)
        if truth is np.nan:
            warning("Ground truth value for parameter " + parameter_name + " was not found!")
        return truth

    # Prior is a parameter
    def get_prior(self, parameter_name):
        parameter = self.get_parameter(parameter_name)
        if parameter is None:
            warning("No probabilistic prior for parameter " + parameter_name + " was found!")
            # TODO: decide if it is a good idea to return this kind of modeler's fixed "prior"...:
            pmean = getattr(self, parameter_name, np.nan)
            if pmean is np.nan:
                pmean = getattr(self.model_config, parameter_name, np.nan)
            if pmean is np.nan:
                warning("No prior value for parameter " + parameter_name + " was found!")
            return pmean, parameter
        else:
            return parameter.mean, parameter

    def get_prior_pdf(self, parameter_name):
        parameter_mean, parameter = self.get_prior(parameter_name)
        if isinstance(parameter, (ProbabilisticParameterBase, TransformedProbabilisticParameterBase)):
            return parameter.scipy_method("pdf")
        else:
            warning("No probabilistic parameter " + parameter_name + " was found!"
                    "\nReturning prior value, if available, instead of pdf!")
            return parameter_mean, np.nan
