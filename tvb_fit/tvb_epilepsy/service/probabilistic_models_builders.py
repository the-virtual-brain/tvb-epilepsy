import time
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy

from tvb_fit.base.constants import PriorsModes, Target_Data_Type
from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_fit.service.probabilistic_parameter_builder\
    import generate_lognormal_parameter, generate_negative_lognormal_parameter, generate_normal_parameter

from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation import calc_eq_z
from tvb_fit.tvb_epilepsy.base.model.epileptor_model_configuration import EpileptorModelConfiguration
from tvb_fit.tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_fit.tvb_epilepsy.base.model.epileptor_probabilistic_models \
    import EpiProbabilisticModel, ODEEpiProbabilisticModel, SDEEpiProbabilisticModel

from tvb_utils.log_error_utils import initialize_logger, warning, raise_value_error
from tvb_utils.data_structures_utils import formal_repr, ensure_list
from tvb_timeseries.model.timeseries import Timeseries as TargetDataTimeseries


x0_def = {"def": X0_DEF, "min": X0_MIN, "max": X0_MAX, }
x1eq_def = {"def": X1EQ_DEF, "min": X1EQ_MIN, "max": X1EQ_MAX}

x_def = {"x0": x0_def, "x1eq": x1eq_def}

DEFAULT_PARAMETERS = [XModes.X0MODE.value, "sigma_"+XModes.X0MODE.value, "K"]
ODE_DEFAULT_PARAMETERS = DEFAULT_PARAMETERS + ["x1_init", "z_init", "epsilon", "scale", "offset"]
SDE_DEFAULT_PARAMETERS = ODE_DEFAULT_PARAMETERS + ["dWt", "sigma"]  # "dX1t", "dZt",


class ProbabilisticModelBuilderBase(object):

    __metaclass__ = ABCMeta

    logger = initialize_logger(__name__)

    model=None
    model_name = "vep"
    model_config = EpileptorModelConfiguration("EpileptorDP2D")
    xmode = XModes.X0MODE.value
    priors_mode = PriorsModes.NONINFORMATIVE.value
    normal_flag = False
    linear_flag = False
    x1eq_cr = X1EQ_CR
    x1eq_def = X1EQ_DEF

    def __init__(self, model=None, model_config=EpileptorModelConfiguration("EpileptorDP2D")):
        if isinstance(model, EpiProbabilisticModel):
            self.model = deepcopy(model)
            self.model_name = self.model.name
            for attr in ["model_config", "normal_flag", "linear_flag", "xmode", "x1eq_cr", "x1eq_def", "priors_mode"]:
                    setattr(self, attr, getattr(self.model, attr))
        else:
            self.model = None
            self.model_name = "vep"
            self.model_config = model_config
            self.xmode = XModes.X1EQMODE.value
            self.priors_mode = PriorsModes.NONINFORMATIVE.value
            self.normal_flag = False
            self.linear_flag = False
            if self.normal_flag:
                self.model_name += "_normal"
            if self.linear_flag:
                self.model_name += "_lin"
            self.x1eq_cr = X1EQ_CR
            self.x1eq_def = X1EQ_DEF

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

    def set_attribute(self, attribute, value):
        setattr(self, attribute, value)
        return self

    def set_attributes(self, attributes_names, attribute_values):
        for attribute_name, attribute_value in zip(ensure_list(attributes_names), ensure_list(attribute_values)):
            self = self.set_attribute(attribute_name, attribute_value)
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

    K = K_DEF
    # MC_direction_split = 0.5
    sigma_x = SIGMA_X0_DEF
    sigma_x_scale = SIGMA_X_SCALE_DEF

    def __init__(self, model=None, model_config=EpileptorModelConfiguration("EpileptorDP2D")): #
        super(ProbabilisticModelBuilder, self).__init__(model,model_config)
        if isinstance(model, EpiProbabilisticModel):
            # self.MC_direction_split = getattr(self.model, "MC_direction_split", self.MC_direction_split)
            self.K = getattr(self.model, "K", self.K)
            self.sigma_x = getattr(self.model, "sigma_x", self.sigma_x)
        else:
            self.K = K_DEF
            # self.MC_direction_split = 0.5
            self.sigma_x = SIGMA_X0_DEF
            self.sigma_x_scale = SIGMA_X_SCALE_DEF

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

    def generate_parameters(self, params_names=DEFAULT_PARAMETERS, parameters=OrderedDict()):
        self.logger.info("Generating model parameters by " + self.__class__.__name__ + "...")
        # Generative model:
        # Epileptor stability:
        if parameters.get(self.xmode, False) is False or (self.xmode in params_names):
            self.logger.info("..." + self.xmode + "...")
            if self.priors_mode == PriorsModes.INFORMATIVE.value:
                xprior = getattr(self.model_config, self.xmode)
                sigma_x = None
            else:
                xprior = x_def[self.xmode]["def"] * np.ones((self.number_of_regions,))
                sigma_x = self.sigma_x
            x_param_name = self.xmode
            parameters.update({self.xmode:
                                   generate_normal_parameter(x_param_name, xprior,
                                                             x_def[self.xmode]["min"], x_def[self.xmode]["max"],
                                                             sigma_x, p_shape=(self.number_of_regions,))})
            # parameters.update({self.xmode: self.generate_normal_or_lognormal_parameter(x_param_name, xprior,
            #                                                                            x_def[self.xmode]["min"],
            #                                                                            x_def[self.xmode]["max"],
            #                                                                            sigma=sigma_x,
            #                                                                            p_shape=(self.number_of_regions,),
            #                                                                            negative_log=True)})

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
                       generate_parameters=True, parameters=OrderedDict(), params_names=DEFAULT_PARAMETERS):
        tic = time.time()
        self.logger.info("Generating model by " + self.__class__.__name__ + "...")
        if generate_parameters:
            parameters = self.generate_parameters(params_names, parameters)
        self.model = EpiProbabilisticModel(self.model_config, self.model_name, target_data_type, self.priors_mode,
                                           int(self.normal_flag), int(self.linear_flag), self.x1eq_cr, self.x1eq_def,
                                           parameters, ground_truth, self.xmode, self.K, self.sigma_x) # , self.MC_direction_split
        self.logger.info(self.__class__.__name__ + " took " +
                         str( time.time() - tic) + ' sec for model generation')
        return self.model


class ODEProbabilisticModelBuilder(ProbabilisticModelBuilder):

    active_regions = np.array([])
    tau1 = TAU1_DEF
    tau0 = TAU0_DEF
    sigma_init = SIGMA_INIT_DEF
    epsilon = EPSILON_DEF
    scale = SCALE_DEF
    offset = OFFSET_DEF
    observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER.value
    projection = np.eye(len(active_regions))
    number_of_target_data = 1
    number_of_seizures = 1
    time_length = SEIZURE_LENGTH
    dt = DT_DEF
    upsample = UPSAMPLE
    x1_prior_weight = 0.0
    x1_scale = 1.0
    x1_offset = 0.0
    def __init__(self, model=None, model_config=EpileptorModelConfiguration("EpileptorDP2D"),
                 number_of_target_data=1, number_of_seizures=1, projection=None):
        super(ODEProbabilisticModelBuilder, self).__init__(model, model_config) # MC_direction_split
        if isinstance(model, ODEEpiProbabilisticModel):
            for attr in ["sigma_init", "tau1", "tau0", "scale", "offset", "epsilon", "x1_scale", "x1_offset",
                         "observation_model", "number_of_target_data", "time_length", "dt", "active_regions",
                         "upsample", "x1_prior_weight", "projection", "number_of_seizures"]:
                setattr(self, attr, getattr(self.model, attr))
        else:
            self.active_regions = np.array(range(self.number_of_regions))
            self.tau1 = TAU1_DEF
            self.tau0 = TAU0_DEF
            self.sigma_init = SIGMA_INIT_DEF
            self.epsilon = EPSILON_DEF
            self.scale = SCALE_DEF
            self.offset = OFFSET_DEF
            self.observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER.value
            self.projection = projection
            if self.projection is None:
                self.projection = np.eye(self.number_of_active_regions)
            self.number_of_target_data = number_of_target_data
            self.number_of_seizures = number_of_seizures
            self.time_length = self.compute_seizure_length()
            self.dt = self.compute_dt()
            self.upsample = self.compute_upsample()
            self.x1_prior_weight = 0.0
            self.x1_scale = 1.0
            self.x1_offset = 0.0

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

    def set_x1_prior_weight(self, x1_prior_weight):
        if (x1_prior_weight >= 0.0 and x1_prior_weight < 1.0):
            self.x1_prior_weight = x1_prior_weight
            return self
        else:
            raise_value_error("x1_prior_weight (%s) is not one inside the interval [0.0, 1.0)!" % str(x1_prior_weight))

    def set_attribute(self, attribute, value):
        if attribute == "x1_prior_weight":
            return self.set_x1_prior_weight(value)
        else:
            return super(ODEProbabilisticModelBuilder, self).set_attribute(attribute, value)

    def compute_seizure_length(self):
        return compute_seizure_length(self.tau0)

    def compute_dt(self):
        return compute_dt(self.tau1)

    def compute_upsample(self, default_seizure_length=SEIZURE_LENGTH):
        return compute_upsample(self.time_length, default_seizure_length, tau0=self.tau0)

    def update_from_timeseries(self, target_data, model_source_ts, x1prior_ts):
        if isinstance(target_data, Timeseries):
            self.number_of_target_data = target_data.number_of_labels
        if isinstance(model_source_ts, Timeseries):
            if self.number_of_active_regions != model_source_ts.number_of_labels:
                warning("Number of active regions doesn't match the number of labels of model's source timeseries!")
        if isinstance(x1prior_ts, Timeseries):
            if self.number_of_active_regions != x1prior_ts.number_of_labels:
                warning("Number of active regions doesn't match the number of labels of x1 source prior timeseries!")

    def generate_parameters(self, params_names=ODE_DEFAULT_PARAMETERS, parameters=OrderedDict(),
                            target_data=None, model_source_ts=None, x1prior_ts=None, x1eps_ts=None):
        parameters = super(ODEProbabilisticModelBuilder, self).generate_parameters(params_names, parameters)
        self.logger.info("Generating model parameters by " + self.__class__.__name__ + "...")
        # if "x1" in params_names:
        #     self.logger.info("...x1...")
        #     n_active_regions = len(active_regions)
        #     if isinstance(model_source_ts, Timeseries) and isinstance(getattr(model_source_ts, "x1", None), Timeseries):
        #         x1_sim_ts = model_source_ts.x1.squeezed
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

        if parameters.get("x1_init", False) is False or "x1_init" in params_names:
            self.logger.info("...x1_init...")
            parameters.update(
                {"x1_init": generate_normal_parameter("x1_init", x1_init, X1_INIT_MIN, X1_INIT_MAX,
                                                      sigma=self.sigma_init, p_shape=(self.number_of_regions,))})

        if parameters.get("z_init", False) is False or "z_init" in params_names:
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
                                                                                         0.0, 5.0 * self.sigma_init,
                                                                                         sigma=self.sigma_init)})

        epsilon_p_shape = ()
        if "epsilon" in params_names or "scale" in params_names or "offset" in params_names:
            if isinstance(model_source_ts, Timeseries) and \
               isinstance(getattr(model_source_ts, "x1", None), Timeseries) and \
               isinstance(target_data, TargetDataTimeseries):
                # model_out_ts = model_source_ts.x1.squeezed[:, self.active_regions] - self.x1eq_def
                # if self.observation_model in OBSERVATION_MODELS.SEEG.value and isinstance(self.projection, np.ndarray):
                #     if self.observation_model == OBSERVATION_MODELS.SEEG_LOGPOWER.value:
                #         model_out_ts = compute_seeg_exp(model_out_ts, self.projection)
                #     else:
                #         model_out_ts = compute_seeg_lin(model_out_ts, self.projection)
                # self.scale = np.max(np.percentile(target_data.data, 99, axis=0) -
                #                     np.percentile(target_data.data, 1, axis=0)) / \
                #              np.max(np.percentile(model_out_ts, 99, axis=0) -
                #                     np.percentile(model_out_ts, 1, axis=0))
                # model_out_ts *= self.scale
                # self.offset = \
                #     (np.percentile(target_data.data, 1, axis=0) - np.percentile(model_out_ts, 1, axis=0)).mean()
                # self.scale = np.max(target_data.data.max(axis=0) - target_data.data.min(axis=0)) / \
                #              np.max(model_out_ts.max(axis=0) - model_out_ts.min(axis=0))
                # self.offset = np.median(target_data.data) - np.median(self.scale*model_out_ts)
                self.epsilon *= np.max(np.percentile(target_data.data, 99, axis=0) -
                                       np.percentile(target_data.data, 1, axis=0))
                # if isinstance(x1eps_ts, Timeseries) and np.all(x1eps_ts.squeezed.shape == target_data.squeezed.shape):
                #     epsilon_p_shape = target_data.shape
                #     self.epsilon *= (1 + zscore(x1eps_ts.squeezed))

            self.logger.info("...observation's model parameters...")

            if "epsilon" in params_names:
                self.logger.info("...epsilon...")
                parameters.update({"epsilon":
                                       self.generate_normal_or_lognormal_parameter("epsilon", self.epsilon,
                                                                                   0, EPSILON_MAX_RATIO * self.epsilon,
                                                                                   sigma=self.epsilon,
                                                                                   p_shape=epsilon_p_shape)})

            if "scale" in params_names:
                self.logger.info("...scale...")
                scale_sigma = self.scale / SCALE_SCALE_DEF
                parameters.update({"scale": self.generate_normal_or_lognormal_parameter("scale", self.scale,
                                                                                        0.75 * self.scale,
                                                                                        1.25 * self.scale,
                                                                                        sigma=scale_sigma)})

            if "offset" in params_names:
                self.logger.info("...offset...")
                offset_sigma = np.maximum(0.1, np.abs(self.offset)/OFFSET_SCALE_DEF)
                parameters.update(
                        {"offset": generate_normal_parameter("offset", self.offset, self.offset - 3 * offset_sigma,
                                                         self.offset + 3 * offset_sigma, sigma=offset_sigma)})

        # if self.x1_prior_weight > 0.0:
        #
        #     if isinstance(x1prior_ts, Timeseries) and \
        #             isinstance(model_source_ts, Timeseries) and \
        #                     isinstance(getattr(model_source_ts, "x1", None), Timeseries):
        #         model_out_ts = model_source_ts.x1.squeezed[:, self.active_regions] # - self.model_config.x1eq.mean() #- X1_MIN #
        #         model_out_ts -= np.mean(model_out_ts, axis=0)
        #         self.offset = np.percentile(x1prior_ts.data, 50) - np.percentile(model_out_ts, 50)   # np.median(self.scale*model_out_ts)
        #         model_out_ts += self.offset
        #         self.scale = np.median(np.percentile(x1prior_ts.data, 99, axis=0) -
        #                                np.percentile(x1prior_ts.data, 1, axis=0)) / \
        #                      np.median(np.percentile(model_out_ts, 99, axis=0) -
        #                                np.percentile(model_out_ts, 1, axis=0))
        #         # self.x1_scale = np.max(x1prior_ts.data.max(axis=0) - x1prior_ts.data.min(axis=0)) / \
        #         #                 np.max(model_out_ts.max(axis=0) - model_out_ts.min(axis=0))
        #         # self.x1_offset = np.median(x1prior_ts.data) - np.median(self.x1_scale * model_out_ts)
        #
        #     if "x1_scale" in params_names:
        #         self.logger.info("...x1_scale...")
        #         x1_scale_scale = self.x1_scale / SCALE_SCALE_DEF
        #         parameters.update({"x1_scale": self.generate_normal_or_lognormal_parameter("x1_scale", self.x1_scale,
        #                                                                                    np.maximum(0.1,
        #                                                                                               self.x1_scale -
        #                                                                                             3 * x1_scale_scale),
        #                                                                                  self.x1_scale +
        #                                                                                    3 * x1_scale_scale,
        #                                                                                   sigma=x1_scale_scale)})
        #     if "x1_offset" in params_names:
        #         self.logger.info("...x1_offset...")
        #         x1_offset_sigma= np.abs(self.x1_offset) / OFFSET_SCALE_DEF
        #         parameters.update(
        #                 {"x1_offset": generate_normal_parameter("x1_offset", self.x1_offset,
        #                                                         self.x1_offset - 3 * x1_offset_sigma,
        #                                                         self.x1_offset + 3 * x1_offset_sigma,
        #                                                         sigma=x1_offset_sigma)})

        return parameters

    def generate_model(self, target_data_type=Target_Data_Type.SYNTHETIC.value, ground_truth={},
                       generate_parameters=True, parameters=OrderedDict(), params_names=ODE_DEFAULT_PARAMETERS,
                       target_data=None, model_source_ts=None, x1prior_ts=None, x1eps_ts=None):
        tic = time.time()
        self.logger.info("Generating model by " + self.__class__.__name__ + "...")
        self.update_from_timeseries(target_data, model_source_ts, x1prior_ts)
        if generate_parameters:
            parameters = self.generate_parameters(params_names, parameters, target_data, model_source_ts, x1prior_ts)
        self.model = ODEEpiProbabilisticModel(self.model_config, self.model_name, target_data_type, self.priors_mode,
                                              int(self.normal_flag), int(self.linear_flag), self.x1eq_cr, self.x1eq_def,
                                              self.x1_prior_weight, parameters, ground_truth, self.xmode,
                                              self.observation_model, self.K, self.sigma_x, self.sigma_init,
                                              self.tau1, self.tau0,
                                              self.epsilon, self.scale, self.offset, self.x1_scale, self.x1_offset,
                                              self.number_of_target_data, self.time_length, self.dt, self.upsample,
                                              self.active_regions, self.projection, self.number_of_seizures)
        self.logger.info(self.__class__.__name__  + " took " +
                         str(time.time() - tic) + ' sec for model generation')
        return self.model


class SDEProbabilisticModelBuilder(ODEProbabilisticModelBuilder):

    sde_mode = SDE_MODES.NONCENTERED.value
    sigma = SIGMA_DEF
    sigma_init=SIGMA_DEF

    def __init__(self, model=None, model_config=EpileptorModelConfiguration("EpileptorDP2D"),
                 number_of_target_data=1, number_of_seizures=1, projection=None):
        super(SDEProbabilisticModelBuilder, self).__init__(model, model_config, number_of_target_data,
                                                           number_of_seizures, projection)
        if isinstance(model, SDEEpiProbabilisticModel):
            for attr in ["sigma_init", "sde_mode", "sigma"]:
                setattr(self, attr, getattr(self.model, attr))
        else:
            self.sde_mode = SDE_MODES.NONCENTERED.value
            self.sigma = SIGMA_DEF
            self.sigma_init = self.sigma

    def _repr(self, d=OrderedDict()):
        d.update(super(SDEProbabilisticModelBuilder, self)._repr(d))
        nKeys = len(d)
        for ikey, (key, val) in enumerate(self.__dict__.items()):
            d.update({str(nKeys+ikey) + ". " + key: str(val)})
        return d

    def generate_parameters(self, params_names=SDE_DEFAULT_PARAMETERS, parameters=OrderedDict(),
                            target_data=None, model_source_ts=None, x1prior_ts=None, x1eps_ts=None):
        parameters = \
            super(SDEProbabilisticModelBuilder, self).generate_parameters(params_names, parameters,
                                                                          target_data, model_source_ts,
                                                                          x1prior_ts, x1eps_ts)
        self.logger.info("Generating model parameters by " + self.__class__.__name__ + "...")
        if "sigma" in params_names:
            self.logger.info("...sigma...")
            parameters.update({"sigma": self.generate_normal_or_lognormal_parameter("sigma", self.sigma,
                                                                                    np.maximum(0.1*self.sigma,
                                                                                               SIGMA_MIN),
                                                                                    np.minimum(3*self.sigma, SIGMA_MAX),
                                                                                    sigma_scale=self.sigma/SIGMA_SCALE)})

        names = []
        mins = []
        maxs = []
        means = []
        if self.sde_mode in [SDE_MODES.CENTERED.value,  SDE_MODES.MIXED.value] \
                or np.any([x in params_names for x in ["x1", "z"]]):
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

        if self.sde_mode in [SDE_MODES.NONCENTERED.value,  SDE_MODES.MIXED.value] or \
                np.any([dWt in params_names for dWt in ["dWt", "dX1t", "dZt"]]):
            self.logger.info("...autoregression noncentered time series parameters...")
            names_dWt = list(set(["dWt", "dX1t", "dZt"]) & set(params_names))  #
            n_xp = len(names_dWt)
            mins += n_xp*[-1.0]
            maxs += n_xp*[1.0]
            means += n_xp*[0.0]
            names += names_dWt

        for iV in range(len(names)):
            self.logger.info("..." + names[iV] + "...")
            if parameters.get(names[iV]) is False or names[iV] in params_names:
                parameters.update(
                    {names[iV]: generate_normal_parameter(names[iV], means[iV],
                                                          mins[iV], maxs[iV], sigma=self.sigma)})

        return parameters

    def generate_model(self, target_data_type=Target_Data_Type.SYNTHETIC.value, ground_truth={},
                       generate_parameters=True, parameters=OrderedDict(), params_names=SDE_DEFAULT_PARAMETERS,
                       target_data=None, model_source_ts=None, x1prior_ts=None, x1eps_ts=None):
        tic = time.time()
        self.logger.info("Generating model by " + self.__class__.__name__ + "...")
        self.update_from_timeseries(target_data, model_source_ts, x1prior_ts)
        if generate_parameters:
            parameters = self.generate_parameters(params_names, parameters, target_data, model_source_ts,
                                                  x1prior_ts, x1eps_ts)
        self.model = SDEEpiProbabilisticModel(self.model_config, self.model_name, target_data_type, self.priors_mode,
                                              int(self.normal_flag), int(self.linear_flag), self.x1eq_cr, self.x1eq_def,
                                              self.x1_prior_weight, parameters, ground_truth, self.xmode,
                                              self.observation_model, self.K, self.sigma_x, self.sigma_init, self.sigma,
                                              self.tau1, self.tau0,
                                              self.epsilon, self.scale, self.offset, self.x1_scale, self.x1_offset,
                                              self.number_of_target_data, self.time_length, self.dt, self.upsample,
                                              self.active_regions, self.projection, self.number_of_seizures,
                                              self.sde_mode)
        self.logger.info(self.__class__.__name__  + " took " +
                         str(time.time() - tic) + ' sec for model generation')
        return self.model
