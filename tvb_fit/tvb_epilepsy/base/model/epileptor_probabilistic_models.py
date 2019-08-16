import numpy as np

from tvb_fit.base.constants import PriorsModes, Target_Data_Type
from tvb_fit.base.model.probabilistic_models.probabilistic_model_base import ProbabilisticModelBase
from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_fit.tvb_epilepsy.base.model.epileptor_model_configuration \
    import EpileptorModelConfiguration as ModelConfiguration

from tvb_utils.log_error_utils import raise_value_error
from tvb_utils.data_structures_utils import ensure_list


class EpiProbabilisticModel(ProbabilisticModelBase):

    name = "vep"
    model_config = ModelConfiguration()
    linear_flag = 0
    normal_flag = 1
    xmode = XModes.X0MODE.value
    x1eq_cr = X1EQ_CR
    x1eq_def = X1EQ_DEF
    sigma_x = SIGMA_X0_DEF
    K = K_DEF
    # MC_direction_split = 0.5

    def __init__(self, model_config=ModelConfiguration(), name='vep', target_data_type=Target_Data_Type.EMPIRICAL.value,
                 priors_mode=PriorsModes.NONINFORMATIVE.value, normal_flag=1,
                 linear_flag=0, x1eq_cr=X1EQ_CR, x1eq_def=X1EQ_DEF,
                 parameters={}, ground_truth={}, xmode=XModes.X0MODE.value,
                 K=K_DEF, sigma_x=SIGMA_X0_DEF):  #, MC_direction_split=0.5
        super(EpiProbabilisticModel, self).__init__(model_config, name, target_data_type, priors_mode, normal_flag,
                                                    parameters, ground_truth)
        self.xmode = xmode
        self.linear_flag = linear_flag
        self.x1eq_cr = x1eq_cr
        self.x1eq_def = x1eq_def
        self.sigma_x = sigma_x
        self.model_config = model_config
        self.K = K
        # self.MC_direction_split = MC_direction_split

    def get_truth(self, parameter_name):
        truth = super(EpiProbabilisticModel, self).get_truth(parameter_name)
        if parameter_name in ["K", "tau1", "tau0"]:
            truth = np.mean(truth)
        return truth

    def get_prior(self, parameter_name):
        pmean, parameter = super(EpiProbabilisticModel, self).get_prior(parameter_name)
        if parameter_name in ["K", "tau1", "tau0"]:
            pmean = np.mean(pmean)
        return pmean, parameter


class ODEEpiProbabilisticModel(EpiProbabilisticModel):

    observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER.value
    x1_prior_weight = 0.0
    sigma_init = SIGMA_INIT_DEF
    tau1 = TAU1_DEF
    tau0 = TAU0_DEF
    scale = 1.0
    offset = 0.0
    epsilon = EPSILON_DEF
    x1_scale = 1.0
    x1_offset = 0.0
    number_of_target_data = 0
    time_length = SEIZURE_LENGTH
    dt = DT_DEF
    upsample = UPSAMPLE
    active_regions = np.array([])
    projection = np.eye(len(active_regions))
    number_of_seizures = 1

    @property
    def nonactive_regions(self):
        return np.delete(np.array(range(self.number_of_regions)), self.active_regions)

    @property
    def number_of_active_regions(self):
        return len(self.active_regions)

    @property
    def number_of_nonactive_regions(self):
        return len(self.nonactive_regions)

    def __init__(self, model_config=ModelConfiguration(), name='vep_ode',
                 target_data_type=Target_Data_Type.EMPIRICAL.value, priors_mode=PriorsModes.NONINFORMATIVE.value,
                 normal_flag=1, linear_flag=0, x1eq_cr=X1EQ_CR, x1eq_def=X1EQ_DEF, x1_prior_weight=0.0,
                 parameters={}, ground_truth={}, xmode=XModes.X0MODE.value,
                 observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER.value,
                 K=K_DEF, sigma_x=SIGMA_X0_DEF, sigma_init=SIGMA_INIT_DEF,
                 tau1=TAU1_DEF, tau0=TAU0_DEF, epsilon=EPSILON_DEF, scale=1.0, offset=0.0, x1_scale=1.0, x1_offset=0.0,
                 number_of_target_data=0, time_length=0, dt=DT_DEF, upsample=UPSAMPLE,
                 active_regions=np.array([]), projection=None, number_of_seizures=1):
        super(ODEEpiProbabilisticModel, self).__init__(model_config, name,  target_data_type, priors_mode, normal_flag,
                                                       linear_flag, x1eq_cr, x1eq_def, parameters, ground_truth,
                                                       xmode, K, sigma_x)
        if np.all(np.in1d(active_regions, range(self.number_of_regions))):
            self.active_regions = np.unique(active_regions)
        else:
            raise_value_error("Active regions indices:\n" + str(active_regions) +
                              "\nbeyond number of regions (" + str(self.number_of_regions) + ")!")

        if observation_model in [_.value for _ in OBSERVATION_MODELS]:
            self.observation_model = observation_model
        else:
            raise_value_error("Statistical model's observation model " + str(observation_model) +
                              " is not one of the valid ones: " + str([_.value for _ in OBSERVATION_MODELS]) + "!")
        self.x1_prior_weight = x1_prior_weight
        self.sigma_init = sigma_init
        self.tau1 = tau1
        self.tau0 = tau0
        self.scale = scale
        self.offset = offset
        self.epsilon = epsilon
        self.x1_scale = x1_scale
        self.x1_offset = x1_offset
        self.number_of_target_data = number_of_target_data
        self.time_length = time_length
        self.dt = dt
        self.upsample = upsample
        self.projection = projection
        if self.projection is None:
            self.projection = np.eye(self.number_of_active_regions)
        self.number_of_seizures =number_of_seizures

    def update_active_regions(self, active_regions):
        if np.all(np.in1d(ensure_list(active_regions), range(self.number_of_regions))):
            self.active_regions = np.unique(ensure_list(active_regions) + self.active_regions.tolist())
        else:
            raise_value_error("Active regions indices:\n" + str(active_regions) +
                              "\nbeyond number of regions (" + str(self.number_of_regions) + ")!")

    def update_seizure_length(self, time_length=None):
        if time_length is None:
            self.time_length = compute_seizure_length(self.tau0)
        else:
            self.time_length = time_length
        return self

    def update_dt(self, dt=None):
        if dt is None:
            self.dt = compute_dt(self.tau1)
        else:
            self.dt = dt
        return self


class SDEEpiProbabilisticModel(ODEEpiProbabilisticModel):

    sigma = SIGMA_DEF
    sde_mode = SDE_MODES.NONCENTERED.value

    def __init__(self, model_config=ModelConfiguration(), name='vep_ode',
                 target_data_type=Target_Data_Type.EMPIRICAL.value, priors_mode=PriorsModes.NONINFORMATIVE.value,
                 normal_flag=1, linear_flag=0, x1eq_cr=X1EQ_CR, x1eq_def=X1EQ_DEF, x1_prior_weight=0.0,
                 parameters={}, ground_truth={}, xmode=XModes.X0MODE.value,
                 observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER.value, K=K_DEF,
                 sigma_x=SIGMA_X0_DEF, sigma_init=SIGMA_INIT_DEF,
                 sigma=SIGMA_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF, epsilon=EPSILON_DEF,
                 scale=1.0, offset=0.0, x1_scale=1.0, x1_offset=0.0,
                 number_of_target_data=0, time_length=0, dt=DT_DEF, upsample=UPSAMPLE, active_regions=np.array([]),
                 projection=None, number_of_seizures=1, sde_mode=SDE_MODES.NONCENTERED.value):
        super(SDEEpiProbabilisticModel, self).__init__(model_config, name, target_data_type, priors_mode, normal_flag,
                                                       linear_flag, x1eq_cr, x1eq_def, x1_prior_weight,
                                                       parameters, ground_truth,
                                                       xmode, observation_model, K,
                                                       sigma_x, sigma_init, tau1, tau0,
                                                       epsilon, scale, offset, x1_scale, x1_offset,
                                                       number_of_target_data, time_length, dt, upsample,
                                                       active_regions, projection, number_of_seizures)
        self.sigma = sigma
        self.sde_mode = sde_mode


class EpileptorProbabilisticModels(Enum):
    EPI_PROBABILISTIC_MODEL = {"name": EpiProbabilisticModel().__class__.__name__,
                               "instance": EpiProbabilisticModel()}
    ODE_EPI_PROBABILISTIC_MODEL = {"name": ODEEpiProbabilisticModel().__class__.__name__,
                                   "instance": ODEEpiProbabilisticModel()}
    SDE_EPI_PROBABILISTIC_MODEL = {"name": SDEEpiProbabilisticModel().__class__.__name__,
                                   "instance": SDEEpiProbabilisticModel()}