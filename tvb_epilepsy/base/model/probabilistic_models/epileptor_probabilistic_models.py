
from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, ensure_list
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.probabilistic_models.probabilistic_parameter import ProbabilisticParameterBase, \
                                                                                TransformedProbabilisticParameterBase

class ProbabilisticModel(object):

    name = "vep"
    target_data_type = TARGET_DATA_TYPE.EMPIRICAL.value
    parameters = {}
    model_config = ModelConfiguration()
    xmode = XModes.X0MODE.value
    priors_mode = PriorsModes.NONINFORMATIVE.value
    sigma_x = SIGMA_X0_DEF
    MC_direction_split = 0.5
    ground_truth = {}

    @property
    def number_of_parameters(self):
        return len(self.parameters)

    def __init__(self, name='vep', number_of_regions=0,  target_data_type=TARGET_DATA_TYPE.EMPIRICAL.value,
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value, parameters={}, ground_truth={},
                 model_config=ModelConfiguration(), sigma_x=SIGMA_X0_DEF, MC_direction_split=0.5):
        self.name = name
        self.number_of_regions = number_of_regions
        self.xmode = xmode
        self.priors_mode = priors_mode
        self.sigma_x = sigma_x
        self.parameters = parameters
        self.model_config = model_config
        self.MC_direction_split = MC_direction_split
        self.ground_truth = ground_truth
        self.target_data_type = target_data_type

    def _repr(self, d=OrderedDict()):
        for ikey, (key, val) in enumerate(self.__dict__.iteritems()):
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
        if self.target_data_type == TARGET_DATA_TYPE.SYNTHETIC.value:
            truth = self.ground_truth.get(parameter_name, np.nan)
            if truth is np.nan:
                truth = getattr(self.model_config, parameter_name, np.nan)
                # TODO: find a more general solution here...
                if truth is np.nan and parameter_name == "MC" or parameter_name == "FC":
                    truth = self.model_config.model_connectivity
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
            return mean_or_truth


class ODEProbabilisticModel(ProbabilisticModel):

    observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER.value
    sigma_init = SIGMA_INIT_DEF
    tau1 = TAU1_DEF
    tau0 = TAU0_DEF
    scale = SCALE_SIGNAL_DEF
    offset = OFFSET_SIGNAL_DEF
    epsilon = EPSILON_DEF
    number_of_target_data = 0
    time_length = 0
    dt = DT_DEF
    active_regions = []

    @property
    def nonactive_regions(self):
        return np.delete(np.array(range(self.number_of_regions)), self.active_regions)

    @property
    def number_of_active_regions(self):
        return len(self.active_regions)

    @property
    def number_of_nonactive_regions(self):
        return len(self.nonactive_regions)

    def __init__(self, name='vep_ode',number_of_regions=0, target_data_type=TARGET_DATA_TYPE.EMPIRICAL.value,
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value, parameters={}, ground_truth={},
                 model_config=ModelConfiguration(), observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER.value,
                 sigma_x=SIGMA_X0_DEF, sigma_init=SIGMA_INIT_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF,
                 scale=SCALE_SIGNAL_DEF, offset=OFFSET_SIGNAL_DEF, epsilon=EPSILON_DEF,
                 number_of_target_data=0, time_length=0, dt=DT_DEF, active_regions=[]):
        super(ODEProbabilisticModel, self).__init__(name, number_of_regions, target_data_type, xmode, priors_mode,
                                                    parameters, ground_truth, model_config, sigma_x)
        if np.all(np.in1d(active_regions, range(self.number_of_regions))):
            self.active_regions = np.unique(active_regions).tolist()
        else:
            raise_value_error("Active regions indices:\n" + str(active_regions) +
                              "\nbeyond number of regions (" + str(self.number_of_regions) + ")!")

        if observation_model in [_.value for _ in OBSERVATION_MODELS]:
            self.observation_model = observation_model
        else:
            raise_value_error("Statistical model's observation model " + str(observation_model) +
                              " is not one of the valid ones: " + str([_.value for _ in OBSERVATION_MODELS]) + "!")
        self.sigma_init = sigma_init
        self.tau1 = tau1
        self.tau0 = tau0
        self.scale = scale
        self.offset = offset
        self.epsilon = epsilon
        self.number_of_target_data = number_of_target_data
        self.time_length = time_length
        self.dt = dt

    def update_active_regions(self, active_regions):
        if np.all(np.in1d(active_regions, range(self.number_of_regions))):
            self.active_regions = np.unique(ensure_list(active_regions) + self.active_regions).tolist()
        else:
            raise_value_error("Active regions indices:\n" + str(active_regions) +
                              "\nbeyond number of regions (" + str(self.number_of_regions) + ")!")


class SDEProbabilisticModel(ODEProbabilisticModel):

    sigma = SIGMA_DEF
    sde_mode = SDE_MODES.NONCENTERED.value

    def __init__(self, name='vep_ode', number_of_regions=0, target_data_type=TARGET_DATA_TYPE.EMPIRICAL.value,
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value, parameters={}, ground_truth={},
                 model_config=ModelConfiguration(), observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER.value,
                 sigma_x=SIGMA_X0_DEF, sigma_init=SIGMA_INIT_DEF, sigma=SIGMA_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF,
                 scale=SCALE_SIGNAL_DEF, offset=OFFSET_SIGNAL_DEF, epsilon=EPSILON_DEF,
                 number_of_target_data=0, time_length=0, dt=DT_DEF, active_regions=[],
                 sde_mode=SDE_MODES.NONCENTERED.value):
        super(SDEProbabilisticModel, self).__init__(name, number_of_regions, target_data_type, xmode, priors_mode,
                                                    parameters, ground_truth, model_config, observation_model,
                                                    sigma_x, sigma_init, tau1, tau0, scale, offset, epsilon,
                                                    number_of_target_data, time_length, dt, active_regions)
        self.sigma = sigma
        self.sde_mode = sde_mode


class EpileptorProbabilisticModels(Enum):
    STATISTICAL_MODEL = ProbabilisticModel().__class__.__name__
    ODESTATISTICAL_MODEL = ODEProbabilisticModel().__class__.__name__
    SDESTATISTICAL_MODEL = SDEProbabilisticModel().__class__.__name__