from enum import Enum

from collections import OrderedDict

from tvb_infer.base.constants import PriorsModes, Target_Data_Type
from tvb_infer.base.utils.log_error_utils import warning, raise_not_implemented_error
from tvb_infer.base.utils.data_structures_utils import formal_repr
from tvb_infer.base.model.probabilistic_models.parameters.transformed_parameters import \
    TransformedProbabilisticParameterBase
from tvb_infer.base.model.probabilistic_models.parameters.base import ProbabilisticParameterBase

class ProbabilisticModelBase(object):

    name = "probabilistic_model"
    number_of_regions = 0
    target_data_type = Target_Data_Type.EMPIRICAL.value
    priors_mode = PriorsModes.NONINFORMATIVE.value
    parameters = {}
    ground_truth = {}

    @property
    def number_of_parameters(self):
        return len(self.parameters)

    def __init__(self, name='vep', number_of_regions=0, target_data_type=Target_Data_Type.EMPIRICAL.value,
                 priors_mode=PriorsModes.NONINFORMATIVE.value, parameters={}, ground_truth={}):
        self.number_of_regions = number_of_regions
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
        if self.target_data_type == Target_Data_Type.SYNTHETIC.value:
            truth = self.ground_truth.get(parameter_name, np.nan)
            if truth is np.nan:
                # TODO: decide if it is a good idea to return this kind of modeler's "truth"...
                truth = getattr(self, parameter_name, np.nan)
            if truth is np.nan:
                warning("Ground truth value for parameter " + parameter_name + " was not found!")
            return truth
        return np.nan

    # Prior is either a parameter or the ground truth
    def get_prior(self, parameter_name):
        parameter = self.get_parameter(parameter_name)
        if parameter is None:
            # TODO: decide if it is a good idea to return this kind of modeler's fixed "prior"...
            return getattr(self, parameter_name, np.nan), None
        else:
            return parameter.mean, parameter

    def get_prior_pdf(self, parameter_name):
        mean_or_truth, parameter = self.get_prior(parameter_name)
        if isinstance(parameter, (ProbabilisticParameterBase, TransformedProbabilisticParameterBase)):
            return parameter.scipy_method("pdf")
        else:
            warning("No parameter " + parameter_name + " was found!\nReturning true value instead of pdf!")
            return mean_or_truth, np.nan


class ProbabilisticModels(Enum):
    raise_not_implemented_error("No probabilitic models available yet!")