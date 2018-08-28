import time
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy

from tvb_fit.base.constants import PriorsModes, Target_Data_Type
from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_fit.base.utils.log_error_utils import initialize_logger, warning
from tvb_fit.base.utils.data_structures_utils import formal_repr, ensure_list
from tvb_fit.service.timeseries_service import compute_seeg_exp, compute_seeg_lin
from tvb_fit.service.probabilistic_parameter_builder\
    import  generate_lognormal_parameter, generate_negative_lognormal_parameter, generate_normal_parameter
from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation import calc_eq_z
from tvb_fit.tvb_epilepsy.base.model.epileptor_model_configuration import EpileptorModelConfiguration
from tvb_fit.tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_fit.tvb_epilepsy.base.model.epileptor_probabilistic_models \
    import EpiProbabilisticModel, ODEEpiProbabilisticModel, SDEEpiProbabilisticModel


x0_def = {"def": X0_DEF, "min": X0_MIN, "max": X0_MAX, }
x1eq_def = {"def": X1EQ_DEF, "min": X1EQ_MIN, "max": X1EQ_MAX}

x_def = {"x0": x0_def, "x1eq": x1eq_def}

DEFAULT_PARAMETERS = [XModes.X0MODE.value, "sigma_"+XModes.X0MODE.value, "K"]
ODE_DEFAULT_PARAMETERS = DEFAULT_PARAMETERS + ["x1_init", "z_init", "epsilon", "scale", "offset"]
SDE_DEFAULT_PARAMETERS =  ODE_DEFAULT_PARAMETERS + ["dX1t", "dZt", "sigma"]


class ProbabilisticModelBuilderBase(object):

    __metaclass__ = ABCMeta

    logger = initialize_logger(__name__)

    model_name = "vep"
    model_config = EpileptorModelConfiguration("EpileptorDP2D")
    xmode = XModes.X0MODE.value
    priors_mode = PriorsModes.NONINFORMATIVE.value
    model = None
    normal_flag = True

    def __init__(self, model=None, model_name="vep", model_config=EpileptorModelConfiguration("EpileptorDP2D"),
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value, normal_flag=True):
        self.model = deepcopy(model)
        self.model_name = model_name
        self.model_config = model_config
        self.xmode = xmode
        self.priors_mode = priors_mode
        self.normal_flag = normal_flag
        if self.normal_flag:
            self.model_name += "_normal"
        if isinstance(self.model, EpiProbabilisticModel):
            self.model_name = self.model.name
            self.model_config = getattr(self.model, "model_config", self.model_config)
            self.xmode = getattr(self.model, "xmode", self.xmode)
            self.priors_mode = getattr(self.model, "priors_mode", self.priors_mode)

    def __repr__(self, d=OrderedDict()):
        return formal_repr(self, self._repr(d))

    def __str__(self):
        return self.__repr__()

    @property
    def number_of_regions(self):
        if isinstance(self.model, EpiProbabilisticModel):
            return self.model.number_of_regions
        else:
            return self.model_config.number_of_regions

    def _repr(self, d=OrderedDict()):
        for ikey, (key, val) in enumerate(self.__dict__.items()):
            d.update({str(ikey) + ". " + key: val})
        return d

    def set_attributes(self, attributes_names, attribute_values):
        for attribute_name, attribute_value in zip(ensure_list(attributes_names), ensure_list(attribute_values)):
            setattr(self, attribute_name, attribute_value)
        return self

    def _set_attributes_from_dict(self, attributes_dict):
        if not isinstance(attributes_dict, dict):
            attributes_dict = attributes_dict.__dict__
        for attr, value in attributes_dict.items():
            if not attr in ["model_config", "parameters", "number_of_regions", "number_of_parameters"]:
                value = attributes_dict.get(attr, None)
                if value is None:
                    warning(attr + " not found in input dictionary!" +
                            "\nLeaving as it is: " + attr + " = " + str(getattr(self, attr)))
                if value is not None:
                    setattr(self, attr, value)
        return attributes_dict

    def generate_normal_or_lognormal_parameter(self, name, mean, low, high, sigma=None,
                                               sigma_scale=2, p_shape=(), use="scipy", negative_log=False):
        if self.normal_flag:
            return generate_normal_parameter(name, mean, low, high, sigma, sigma_scale, p_shape, use)
        else:
            if negative_log:
                return generate_negative_lognormal_parameter(name, mean, low, high, sigma, sigma_scale, p_shape, use)
            else:
                return generate_lognormal_parameter(name, mean, low, high, sigma, sigma_scale, p_shape, use)

    @abstractmethod
    def generate_parameters(self):
        pass

    @abstractmethod
    def generate_model(self):
        pass


class ProbabilisticModelBuilder(ProbabilisticModelBuilderBase):

    sigma_x = SIGMA_X0_DEF
    sigma_x_scale = 3
    K = K_DEF
    # MC_direction_split = 0.5

    def __init__(self, model=None, model_name="vep", model_config=EpileptorModelConfiguration("EpileptorDP2D"),
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value, normal_flag=True,
                 K=K_DEF, sigma_x=None, sigma_x_scale=3): #
        super(ProbabilisticModelBuilder, self).__init__(model, model_name, model_config,
                                                        xmode, priors_mode, normal_flag)
        self.K = K
        # self.MC_direction_split = MC_direction_split
        if sigma_x is None:
            if self.xmode == XModes.X0MODE.value:
                self.sigma_x = SIGMA_X0_DEF
            else:
                self.sigma_x = SIGMA_EQ_DEF
        else:
            self.sigma_x = sigma_x
        self.sigma_x_scale = sigma_x_scale
        if isinstance(self.model, EpiProbabilisticModel):
            # self.MC_direction_split = getattr(self.model, "MC_direction_split", self.MC_direction_split)
            self.K = getattr(self.model, "K", self.K)
            self.sigma_x = getattr(self.model, "sigma_x", self.sigma_x)

    def _repr(self, d=OrderedDict()):
        d.update(super(ProbabilisticModelBuilder, self)._repr(d))
        nKeys = len(d)
        for ikey, (key, val) in enumerate(self.__dict__.items()):
            d.update({str(nKeys+ikey) + ". " + key: str(val)})
        return d

    def get_SC(self, model_connectivity):
        # Set symmetric connectivity to be in the interval [MC_MAX / MAX_MIN_RATIO, MC_MAX],
        # where self.MC_MAX corresponds to the 95th percentile of connectivity
        p95 = np.percentile(model_connectivity.flatten(), 95)
        SC = np.array(model_connectivity)
        if p95 != MC_MAX:
            SC = SC / p95
            SC[SC > MC_MAX] = 1.0
        mc_def_min = MC_MAX / MC_MAX_MIN_RATIO
        SC[SC < mc_def_min] = mc_def_min
        diag_ind = range(self.number_of_regions)
        SC[diag_ind, diag_ind] = 0.0
        return SC

    def get_MC_prior(self, model_connectivity):
        MC_def = self.get_SC(model_connectivity)
        # inds = np.triu_indices(self.number_of_regions, 1)
        # MC_def[inds] = MC_def[inds] * self.MC_direction_split
        # MC_def = MC_def.T
        # MC_def[inds] = MC_def[inds] * (1.0 - self.MC_direction_split)
        # MC_def = MC_def.T
        MC_def[MC_def < 0.001] = 0.001
        return MC_def

    def generate_parameters(self, params_names=DEFAULT_PARAMETERS):
        parameters = OrderedDict()
        self.logger.info("Generating model parameters by " + self.__class__.__name__ + "...")
        # Generative model:
        # Epileptor stability:
        self.logger.info("..." + self.xmode + "...")
        if self.priors_mode == PriorsModes.INFORMATIVE.value:
            xprior = getattr(self.model_config, self.xmode)
            sigma_x = None
        else:
            xprior = x_def[self.xmode]["def"] * np.ones((self.number_of_regions,))
            sigma_x = self.sigma_x
        x_param_name = self.xmode
        parameters.update({self.xmode: self.generate_normal_or_lognormal_parameter(x_param_name, xprior,
                                                                                   x_def[self.xmode]["min"],
                                                                                   x_def[self.xmode]["max"],
                                                                                   sigma=sigma_x,
                                                                                   p_shape=(self.number_of_regions,),
                                                                                   negative_log=True)})

        # Update sigma_x value and name
        self.sigma_x = parameters[self.xmode].std
        sigma_x_name = "sigma_" + self.xmode
        if "sigma_x" in params_names:
            self.logger.info("...sigma_x...")
            parameters.update({"sigma_x": self.generate_normal_or_lognormal_parameter(sigma_x_name, self.sigma_x,
                                                                                      0.0, 10 * self.sigma_x,
                                                                                      sigma_scale=self.sigma_x_scale)})


        # Coupling
        if "MC" in params_names:
            self.logger.info("...MC...")
            parameters.update({"MC": self.generate_normal_or_lognormal_parameter("MC",
                                                                      self.get_MC_prior(self.model_config.connectivity),
                                                                                 MC_MIN, MC_MAX, sigma_scale=MC_SCALE)})

        if "K" in params_names:
            self.logger.info("...K...")
            parameters.update({"K": self.generate_normal_or_lognormal_parameter("K", self.K, K_MIN, K_MAX,
                                                                                sigma_scale=K_SCALE)})

        return parameters

    def generate_model(self, target_data_type=Target_Data_Type.SYNTHETIC.value, ground_truth={},
                       generate_parameters=True, params_names=DEFAULT_PARAMETERS):
        tic = time.time()
        self.logger.info("Generating model by " + self.__class__.__name__ + "...")
        if generate_parameters:
            parameters = self.generate_parameters(params_names)
        else:
            parameters = {}
        self.model = EpiProbabilisticModel(self.model_config, self.model_name, target_data_type, self.priors_mode,
                                           parameters, ground_truth, self.xmode,
                                           self.K, self.sigma_x) # , self.MC_direction_split
        self.logger.info(self.__class__.__name__ + " took " +
                         str( time.time() - tic) + ' sec for model generation')
        return self.model


class ODEProbabilisticModelBuilder(ProbabilisticModelBuilder):

    sigma_init = SIGMA_INIT_DEF
    epsilon = EPSILON_DEF
    scale = SCALE_DEF
    offset = OFFSET_DEF
    observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER.value
    number_of_target_data = 0
    active_regions = np.array([])
    tau1 = TAU1_DEF
    tau0 = TAU0_DEF
    time_length = SEIZURE_LENGTH
    dt = DT_DEF

    def __init__(self, model=None, model_name="vep_ode", model_config=EpileptorModelConfiguration("EpileptorDP2D"),
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value, normal_flag=True,
                 K=K_DEF, sigma_x=None, sigma_x_scale=3,  # MC_direction_split=0.5,
                 sigma_init=SIGMA_INIT_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF,
                 epsilon=EPSILON_DEF, scale=SCALE_DEF, offset=OFFSET_DEF,
                 observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER.value,
                 number_of_target_data=0, active_regions=np.array([])):
        super(ODEProbabilisticModelBuilder, self).__init__(model, model_name, model_config, xmode, priors_mode,
                                                           normal_flag, K, sigma_x, sigma_x_scale) # MC_direction_split
        self.sigma_init = sigma_init
        self.tau1 = tau1
        self.tau0 = tau0
        self.epsilon = epsilon
        self.scale = scale
        self.offset = offset
        self.observation_model = observation_model
        self.number_of_target_data = number_of_target_data
        self.time_length = self.compute_seizure_length()
        self.dt = self.compute_dt()
        self.active_regions = active_regions
        if isinstance(self.model, EpiProbabilisticModel):
            self.sigma_init = getattr(self.model, "sigma_init", self.sigma_init)
            self.tau1 = getattr(self.model, "tau1", self.tau1)
            self.tau0 = getattr(self.model, "tau0", self.tau0)
            self.scale = getattr(self.model, "scale", self.scale)
            self.offset = getattr(self.model, "offset", self.offset)
            self.epsilon = getattr(self.model, "epsilon", self.epsilon)
            self.observation_model = getattr(self.model, "observation_model", self.observation_model)
            self.number_of_target_data = getattr(self.model, "number_of_target_data", self.number_of_target_data)
            self.time_length = getattr(self.model, "time_length", self.time_length)
            self.dt = getattr(self.model, "dt", self.dt)
            self.active_regions = getattr(self.model, "active_regions", self.active_regions)

    def _repr(self, d=OrderedDict()):
        d.update(super(ODEProbabilisticModelBuilder, self)._repr(d))
        nKeys = len(d)
        for ikey, (key, val) in enumerate(self.__dict__.items()):
            d.update({str(nKeys+ikey) + ". " + key: str(val)})
        return d

    @property
    def number_of_active_regions(self):
        if isinstance(self.model, ODEEpiProbabilisticModel):
            return len(self.model.active_regions)
        else:
            return len(self.active_regions)

    @property
    def get_active_regions(self):
        if isinstance(self.model, ODEEpiProbabilisticModel):
            return self.model.active_regions
        else:
            return self.active_regions

    @property
    def get_number_of_target_data(self):
        if isinstance(self.model, ODEEpiProbabilisticModel):
            return self.model.number_of_target_data
        else:
            return self.number_of_target_data

    @property
    def get_time_length(self):
        if isinstance(self.model, ODEEpiProbabilisticModel):
            return self.model.time_length
        else:
            return self.time_length

    def compute_seizure_length(self):
        return compute_seizure_length(self.tau0)

    def compute_dt(self):
        return compute_dt(self.tau1)

    def generate_parameters(self, params_names=ODE_DEFAULT_PARAMETERS,
                            target_data=None, source_ts=None, gain_matrix=None):
        parameters = super(ODEProbabilisticModelBuilder, self).generate_parameters(params_names)
        if isinstance(self.model, ODEEpiProbabilisticModel):
            active_regions = self.model.active_regions
        else:
            active_regions =  np.array([])
        if len(active_regions) == 0:
            active_regions = np.array(range(self.number_of_regions))
        self.logger.info("Generating model parameters by " + self.__class__.__name__ + "...")
        # if "x1" in params_names:
        #     self.logger.info("...x1...")
        #     n_active_regions = len(active_regions)
        #     if isinstance(source_ts, Timeseries) and isinstance(getattr(source_ts, "x1", None), Timeseries):
        #         x1_sim_ts = source_ts.x1.squeezed
        #         mu_prior = np.zeros(n_active_regions, )
        #         sigma_prior = np.zeros(n_active_regions, )
        #         loc_prior = np.zeros(n_active_regions, )
        #         if self.priors_mode == PriorsModes.INFORMATIVE.value:
        #             for ii, iR in enumerate(active_regions):
        #                 fit = ss.lognorm.fit(x1_sim_ts[:, iR] - X1_MIN)
        #                 sigma_prior[ii] = fit[0]
        #                 mu_prior[ii] = np.log(fit[2]) # mu = exp(scale)
        #                 loc_prior[ii] = fit[1] + X1_MIN
        #         else:
        #             fit = ss.lognorm.fit(x1_sim_ts[:, active_regions].flatten() - X1_MIN)
        #             sigma_prior += fit[0]
        #             mu_prior += np.log(fit[2])  # mu = exp(scale)
        #             loc_prior += fit[1] + X1_MIN
        #     else:
        #         sigma_prior = X1_LOGSIGMA_DEF * np.ones(n_active_regions, )
        #         mu_prior = X1_LOGMU_DEF * np.ones(n_active_regions, )
        #         loc_prior = X1_LOGLOC_DEF * np.ones(n_active_regions, ) + X1_MIN
        #         if self.priors_mode == PriorsModes.INFORMATIVE.value:
        #             sigma_prior[self.active_regions] = X1_LOGSIGMA_ACTIVE
        #             mu_prior[self.active_regions] = X1_LOGMU_ACTIVE
        #             loc_prior[self.active_regions] = X1_LOGLOC_ACTIVE + X1_MIN
        #     parameters.update(
        #         {"x1":
        #              generate_probabilistic_parameter("x1", X1_MIN, X1_MAX, p_shape=(n_active_regions, ),
        #                                               probability_distribution=ProbabilityDistributionTypes.LOGNORMAL,
        #                                               optimize_pdf=False, use="scipy", loc=loc_prior,
        #                                               **{"mu": mu_prior, "sigma": sigma_prior})})
        self.logger.info("...initial conditions' parameters...")
        if self.priors_mode == PriorsModes.INFORMATIVE.value:
            x1_init = self.model_config.x1eq
            z_init = self.model_config.zeq
        else:
            x1_init = X1_REST * np.ones((self.number_of_regions,))
            z_init = calc_eq_z(x1_init, self.model_config.yc, self.model_config.Iext1, "2d", x2=0.0,
                               slope=self.model_config.slope, a=self.model_config.a, b=self.model_config.b,
                               d=self.model_config.d, x1_neg=True)
        self.logger.info("...x1_init...")
        parameters.update(
            {"x1_init": generate_normal_parameter("x1_init", x1_init, X1_INIT_MIN, X1_INIT_MAX,
                                                  sigma=self.sigma_init, p_shape=(self.number_of_regions,))})
        self.logger.info("...z_init...")
        parameters.update(
            {"z_init": generate_normal_parameter("z_init", z_init, Z_INIT_MIN, Z_INIT_MAX,
                                                  sigma=self.sigma_init/2.0, p_shape=(self.number_of_regions,))})

        # Time scales
        if "tau1" in params_names:
            self.logger.info("...tau1...")
            parameters.update({"tau1": self.generate_normal_or_lognormal_parameter("tau1", self.tau1,
                                                                                   TAU1_MIN, TAU1_MAX,
                                                                                   sigma_scale=TAU1_SCALE)})

        if "tau0" in params_names:
            self.logger.info("...tau0...")
            parameters.update({"tau0": self.generate_normal_or_lognormal_parameter("tau0", self.tau0,
                                                                                   TAU0_MIN, TAU0_MAX,
                                                                                   sigma_scale=TAU0_SCALE)})

        if "sigma_init" in params_names:
            self.logger.info("...sigma_init...")
            parameters.update({"sigma_init": self.generate_normal_or_lognormal_parameter("sigma_init", self.sigma_init,
                                                                                         0.0, 10.0 * self.sigma_init,
                                                                                         sigma=self.sigma_init)})

        self.logger.info("...observation's model parameters...")
        if "epsilon" in params_names:
            self.logger.info("...epsilon...")
            parameters.update({"epsilon": self.generate_normal_or_lognormal_parameter("epsilon", self.epsilon,
                                                                                      0.0, 10.0 * self.epsilon,
                                                                                      sigma=self.epsilon)})

        if isinstance(source_ts, Timeseries) and isinstance(getattr(source_ts, "x1", None), Timeseries) and \
            isinstance(target_data, Timeseries):
            sim_seeg = source_ts.x1.squeezed[:, active_regions] - self.model_config.x1eq.mean()
            if isinstance(gain_matrix, np.ndarray):
                if self.observation_model == OBSERVATION_MODELS.SEEG_LOGPOWER.value:
                    sim_seeg = compute_seeg_exp(sim_seeg, gain_matrix)
                else:
                    sim_seeg = compute_seeg_lin(sim_seeg, gain_matrix)
            self.scale = target_data.data.std() / sim_seeg.std()
            self.offset = np.median(target_data.data) - np.median(self.scale*sim_seeg)

        if "scale" in params_names:
            self.logger.info("...scale...")
            parameters.update({"scale": self.generate_normal_or_lognormal_parameter("scale", self.scale,
                                                                                    np.minimum(0.1, self.scale/3),
                                                                                    3.0 * self.scale,
                                                                                    sigma=self.scale)})
            
        if "offset" in params_names:
            self.logger.info("...offset...")
            parameters.update({"offset": generate_normal_parameter("offset", self.offset,
                                                                   self.offset-1.0, self.offset + 1.0, sigma=1.0)})

        return parameters

    def generate_model(self, target_data_type=Target_Data_Type.SYNTHETIC.value, ground_truth={},
                       generate_parameters=True, params_names=ODE_DEFAULT_PARAMETERS,
                       target_data=None, sim_signals=None, gain_matrix=None):
        tic = time.time()
        self.logger.info("Generating model by " + self.__class__.__name__ + "...")
        if generate_parameters:
            parameters = self.generate_parameters(params_names, target_data, sim_signals, gain_matrix)
        else:
            parameters = {}
        self.model = ODEEpiProbabilisticModel(self.model_config, self.model_name, target_data_type, self.priors_mode,
                                              parameters, ground_truth, self.xmode, self.observation_model,
                                              self.K, self.sigma_x, self.sigma_init, self.tau1, self.tau0,
                                              self.epsilon, self.scale, self.offset, self.number_of_target_data,
                                              self.time_length, self.dt, self.active_regions)
        self.logger.info(self.__class__.__name__  + " took " +
                         str(time.time() - tic) + ' sec for model generation')
        return self.model


class SDEProbabilisticModelBuilder(ODEProbabilisticModelBuilder):

    sigma = SIGMA_DEF
    sde_mode = SDE_MODES.NONCENTERED.value

    def __init__(self, model=None, model_name="vep_sde", model_config=EpileptorModelConfiguration("EpileptorDP2D"),
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value, normal_flag=True,
                 K=K_DEF, sigma_x=None, sigma_x_scale=3,  # MC_direction_split=0.5,
                 sigma_init=SIGMA_INIT_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF,
                 epsilon=EPSILON_DEF, scale=SCALE_DEF, offset=OFFSET_DEF, sigma=SIGMA_DEF,
                 sde_mode=SDE_MODES.NONCENTERED.value, observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER.value,
                 number_of_signals=0, active_regions=np.array([])):
        super(SDEProbabilisticModelBuilder, self).__init__(model, model_name, model_config, xmode, priors_mode,
                                                           normal_flag, K, sigma_x, sigma_x_scale, # MC_direction_split,
                                                           sigma_init, tau1, tau0, epsilon, scale, offset,
                                                           observation_model, number_of_signals, active_regions)
        self.sigma_init = sigma_init
        self.sde_mode = sde_mode
        self.sigma = sigma
        if isinstance(self.model, SDEEpiProbabilisticModel):
            self.sigma_init = getattr(self.model, "sigma_init", self.sigma_init)
            self.sde_mode = getattr(self.model, "sde_mode", self.sde_mode)
            self.sigma = getattr(self.model, "sigma", self.sigma)

    def _repr(self, d=OrderedDict()):
        d.update(super(SDEProbabilisticModelBuilder, self)._repr(d))
        nKeys = len(d)
        for ikey, (key, val) in enumerate(self.__dict__.items()):
            d.update({str(nKeys+ikey) + ". " + key: str(val)})
        return d

    def generate_parameters(self, params_names=SDE_DEFAULT_PARAMETERS, 
                            target_data=None, source_ts=None, gain_matrix=None):
        parameters = \
            super(SDEProbabilisticModelBuilder, self).generate_parameters(params_names,
                                                                          target_data, source_ts, gain_matrix)
        self.logger.info("Generating model parameters by " + self.__class__.__name__ + "...")
        if "sigma" in params_names:
            self.logger.info("...sigma...")
            parameters.update({"sigma": self.generate_normal_or_lognormal_parameter("sigma", self.sigma,
                                                                                    SIGMA_MIN, SIGMA_MAX,
                                                                                    sigma_scale=SIGMA_SCALE)})

        names = []
        mins = []
        maxs = []
        means = []
        if self.sde_mode == SDE_MODES.CENTERED.value:
            self.logger.info("...autoregression centered time series parameters...")
            if "x1" in params_names:
                names.append("x1")
                mins.append(X1_MIN)
                maxs.append(X1_MAX)
                means.append(X1_REST)
            if "z" in params_names:
                names.append("z")
                mins.append(Z_MIN)
                maxs.append(Z_MAX)
                means.append(calc_eq_z(X1_REST, self.model_config.yc, self.model_config.Iext1, "2d", x2=0.0,
                               slope=self.model_config.slope, a=self.model_config.a, b=self.model_config.b,
                               d=self.model_config.d, x1_neg=True))
            n_xp =len(names)
        else:
            self.logger.info("...autoregression noncentered time series parameters...")
            names = list(set(["dX1t", "dZt"]) & set(params_names))
            n_xp = len(names)
            mins = n_xp*[-1.0]
            maxs = n_xp*[1.0]
            means = n_xp*[0.0]
        for iV in range(n_xp):
            self.logger.info("..." + names[iV] + "...")
            parameters.update(
                {names[iV]: generate_normal_parameter(names[iV], means[iV],
                                                      mins[iV], maxs[iV], sigma=self.sigma)})

        return parameters

    def generate_model(self, target_data_type=Target_Data_Type.SYNTHETIC.value, ground_truth={},
                       generate_parameters=True, params_names=SDE_DEFAULT_PARAMETERS,
                       target_data=None, sim_signals=None, gain_matrix=None):
        tic = time.time()
        self.logger.info("Generating model by " + self.__class__.__name__ + "...")
        if generate_parameters:
            parameters = self.generate_parameters(params_names, target_data, sim_signals, gain_matrix)
        else:
            parameters = {}
        self.model = SDEEpiProbabilisticModel(self.model_config, self.model_name, target_data_type, self.priors_mode,
                                              parameters, ground_truth, self.xmode,self.observation_model, self.K,
                                              self.sigma_x, self.sigma_init, self.sigma, self.tau1, self.tau0,
                                              self.epsilon, self.scale, self.offset,self.number_of_target_data,
                                              self.time_length, self.dt, self.active_regions, self.sde_mode)
        self.logger.info(self.__class__.__name__  + " took " +
                         str(time.time() - tic) + ' sec for model generation')
        return self.model
