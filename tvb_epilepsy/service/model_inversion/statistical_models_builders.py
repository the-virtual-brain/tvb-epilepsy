import time
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np
from tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, ensure_list
from tvb_epilepsy.base.computations.equilibrium_computation import calc_eq_z
from tvb_epilepsy.base.computations.probability_distributions import ProbabilityDistributionTypes
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.statistical_models.epileptor_statistical_models \
                                                     import StatisticalModel, ODEStatisticalModel, SDEStatisticalModel
from tvb_epilepsy.service.stochastic_parameter_builder import generate_stochastic_parameter
from tvb_epilepsy.service.model_inversion.epileptor_params_factory \
                                            import generate_lognormal_parameter, generate_negative_lognormal_parameter

x0_def = {"def": X0_DEF, "min": X0_MIN, "max": X0_MAX, }
x1eq_def = {"def": X1EQ_DEF, "min": X1EQ_MIN, "max": X1EQ_MAX}

x_def = {"x0": x0_def, "x1eq": x0_def}


class StatisticalModelBuilderBase(object):

    __metaclass__ = ABCMeta

    logger = initialize_logger(__name__)

    name = "vep"
    model_config = ModelConfiguration()
    parameters = [XModes.X0MODE.value]
    xmode = XModes.X0MODE.value
    priors_mode = PriorsModes.NONINFORMATIVE.value

    def __init__(self, model_name="vep", model_config=ModelConfiguration(), parameters=[XModes.X0MODE.value],
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value):
        self.name = model_name
        self.model_config = model_config
        self.xmode = xmode
        self.parameters = parameters
        self.priors_mode = priors_mode

    def __repr__(self, d=OrderedDict()):
        return formal_repr(self, self._repr(d))

    def __str__(self):
        return self.__repr__()

    @property
    def number_of_regions(self):
        return self.model_config.number_of_regions

    @property
    def number_of_parameters(self):
        return len(self.parameters)

    def _repr(self, d=OrderedDict()):
        for ikey, (key, val) in enumerate(self.__dict__.iteritems()):
            d.update({str(ikey) + ". " + key: val})
        return d

    def set_attributes(self, attributes_names, attribute_values):
        for attribute_name, attribute_value in zip(ensure_list(attributes_names), ensure_list(attribute_values)):
            setattr(self, attribute_name, attribute_value)
        return self

    def _set_attributes_from_dict(self, attributes_dict):
        if not isinstance(attributes_dict, dict):
            attributes_dict = attributes_dict.__dict__
        for attr, value in attributes_dict.iteritems():
            if not attr in ["model_config", "parameters", "number_of_regions", "number_of_parameters"]:
                value = attributes_dict.get(attr, None)
                if value is None:
                    warning(attr + " not found in input dictionary!" +
                            "\nLeaving as it is: " + attr + " = " + str(getattr(self, attr)))
                if value is not None:
                    setattr(self, attr, value)
        return attributes_dict

    @abstractmethod
    def generate_parameters(self):
        pass

    @abstractmethod
    def generate_model(self):
        pass


class StatisticalModelBuilder(StatisticalModelBuilderBase):

    parameters = [XModes.X0MODE.value, "sigma_"+XModes.X0MODE.value, "K"]
    sigma_x = SIGMA_X0_DEF
    sigma_x_scale = 3
    MC_direction_split = 0.5
    model_config = ModelConfiguration()

    def __init__(self, model_name="vep", model_config=ModelConfiguration(),
                 parameters=[XModes.X0MODE.value, "sigma_"+XModes.X0MODE.value, "K"],
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value,
                 sigma_x=None, sigma_x_scale=3, MC_direction_split=0.5):
        super(StatisticalModelBuilder, self).__init__(model_name, model_config, parameters, xmode, priors_mode)
        if sigma_x is None:
            if self.xmode == XModes.X0MODE.value:
                self.sigma_x = SIGMA_X0_DEF
            else:
                self.sigma_x = SIGMA_EQ_DEF
        else:
            self.sigma_x = sigma_x
        self.sigma_x_scale = sigma_x_scale
        self.MC_direction_split = MC_direction_split

    def _repr(self, d=OrderedDict()):
        d.update(super(StatisticalModelBuilder, self)._repr(d))
        nKeys = len(d)
        for ikey, (key, val) in enumerate(self.__dict__.iteritems()):
            d.update({str(nKeys+ikey) + ". " + key: str(val)})
        return d

    def get_SC(self, model_connectivity):
        # Set symmetric connectivity to be in the interval [MC_MAX / MAX_MIN_RATIO, MC_MAX],
        # where self.MC_MAX corresponds to the 95th percentile of model_connectivity
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
        inds = np.triu_indices(self.number_of_regions, 1)
        MC_def[inds] = MC_def[inds] * self.MC_direction_split
        MC_def = MC_def.T
        MC_def[inds] = MC_def[inds] * (1.0 - self.MC_direction_split)
        MC_def = MC_def.T
        MC_def[MC_def < 0.001] = 0.001
        return MC_def

    def generate_parameters(self):
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
        parameters.update(
            {self.xmode: generate_negative_lognormal_parameter(x_param_name, xprior,
                                                               x_def[self.xmode]["min"],
                                                               x_def[self.xmode]["max"],
                                                               sigma=sigma_x, sigma_scale=self.sigma_x_scale,
                                                               p_shape=(self.number_of_regions,), use="scipy")})
        # Update sigma_x value and name
        self.sigma_x = parameters[self.xmode].std
        sigma_x_name = "sigma_" + self.xmode
        if sigma_x in self.parameters:
            self.logger.info("..." + sigma_x + "...")
            parameters.update(
                {sigma_x: generate_lognormal_parameter(sigma_x_name, self.sigma_x, 0.0, 10*self.sigma_x,
                                                       sigma=self.sigma_x, sigma_scale=10, p_shape=(), use="scipy")})

        # Coupling
        if "MC" in self.parameters:
            self.logger.info("...MC...")
            parameters.update(
                {"MC": generate_lognormal_parameter("MC", self.get_MC_prior(self.model_config.model_connectivity),
                                                    MC_MIN, MC_MAX, sigma=None, sigma_scale=MC_SCALE,
                                                    p_shape=(), use="scipy")})

        if "K" in self.parameters:
            self.logger.info("...K...")
            parameters.update(
                {"K": generate_lognormal_parameter("K", self.model_config.K, K_MIN, K_MAX, sigma=None,
                                                   sigma_scale=K_SCALE, p_shape=(), use="scipy")})

        return parameters

    def generate_model(self, target_data_type=TARGET_DATA_TYPE.SYNTHETIC.value, ground_truth={}):
        tic = time.time()
        self.logger.info("Generating model by " + self.__class__.__name__ + "...")
        parameters = self.generate_parameters()
        model = StatisticalModel(self.name, self.number_of_regions, target_data_type, self.xmode, self.priors_mode,
                                 parameters, ground_truth, self.model_config, self.sigma_x, self.MC_direction_split)
        self.logger.info(self.__class__.__name__  + " took " +
                         str( time.time() - tic) + ' sec for model generation')
        return model



class ODEStatisticalModelBuilder(StatisticalModelBuilder):

    parameters = [XModes.X0MODE.value, "sigma_"+XModes.X0MODE.value, "tau1", "K", "x1init", "zinit",
                  "sigma_init", "epsilon", "scale", "offset"]
    sigma_init = SIGMA_INIT_DEF
    epsilon = EPSILON_DEF
    scale = SCALE_SIGNAL_DEF
    offset = OFFSET_SIGNAL_DEF
    observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER.value
    number_of_signals = 0
    time_length = 0
    dt = DT_DEF
    active_regions = []
    tau1 = TAU1_DEF
    tau0 = TAU0_DEF

    def __init__(self, model_name="vep_ode", model_config=ModelConfiguration(),
                 parameters=[XModes.X0MODE.value, "sigma_"+XModes.X0MODE.value, "tau1", "K", "x1init", "zinit",
                             "sigma_init", "epsilon", "scale", "offset"],
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value,
                 sigma_x=None, sigma_x_scale=3, MC_direction_split=0.5,
                 sigma_init=SIGMA_INIT_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF, epsilon=EPSILON_DEF,
                 scale=SCALE_SIGNAL_DEF, offset=OFFSET_SIGNAL_DEF,
                 observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER.value,
                 number_of_signals=0, time_length=0, dt=DT_DEF, active_regions=[]):
        super(ODEStatisticalModelBuilder, self).__init__(model_name, model_config, parameters, xmode, priors_mode,
                                                         sigma_x, sigma_x_scale, MC_direction_split)
        self.sigma_init = sigma_init
        self.tau1 = tau1
        self.tau0 = tau0
        self.epsilon = epsilon
        self.scale = scale
        self.offset = offset
        self.observation_model = observation_model
        self.number_of_signals = number_of_signals
        self.time_length = time_length
        self.dt = dt
        self.active_regions = active_regions

    def _repr(self, d=OrderedDict()):
        d.update(super(ODEStatisticalModelBuilder, self)._repr(d))
        nKeys = len(d)
        for ikey, (key, val) in enumerate(self.__dict__.iteritems()):
            d.update({str(nKeys+ikey) + ". " + key: str(val)})
        return d

    def generate_parameters(self):
        parameters = super(ODEStatisticalModelBuilder, self).generate_parameters()
        self.logger.info("Generating model parameters by " + self.__class__.__name__ + "...")

        self.logger.info("...initial conditions' parameters...")
        if self.priors_mode == PriorsModes.INFORMATIVE.value:
            x1init = self.model_config.x1eq
            zinit = self.model_config.zeq
        else:
            x1init = X1_REST * np.ones((self.number_of_regions,))
            zinit = calc_eq_z(x1init, self.model_config.yc, self.model_config.Iext1, "2d", x2=0.0,
                              slope=self.model_config.slope, a=self.model_config.a, b=self.model_config.b,
                              d=self.model_config.d, x1_neg=True)
        self.logger.info("...x1init...")
        parameters.update(
            {"x1init": generate_stochastic_parameter("x1init", X1_MIN, X1_MAX,
                                                     p_shape=(self.number_of_regions,),
                                                     probability_distribution=ProbabilityDistributionTypes.NORMAL,
                                                     optimize_pdf=False, use="scipy",
                                                     **{"mu": x1init, "sigma": self.sigma_init})})
        self.logger.info("...zinit...")
        parameters.update(
            {"zinit": generate_stochastic_parameter("zinit", Z_MIN, Z_MAX,
                                                     p_shape=(self.number_of_regions,),
                                                     probability_distribution=ProbabilityDistributionTypes.NORMAL,
                                                     optimize_pdf=False, use="scipy",
                                                     **{"mu": zinit, "sigma": self.sigma_init/2})})

        # Time scales
        if "tau1" in self.parameters:
            self.logger.info("...tau1...")
            parameters.update(
                {"tau1": generate_lognormal_parameter("tau1", self.tau1, TAU1_MIN, TAU1_MAX, sigma=None,
                                                      sigma_scale=TAU1_SCALE, p_shape=(), use="scipy")})

        if "tau0" in self.parameters:
            self.logger.info("...tau0...")
            parameters.update(
                {"tau0": generate_lognormal_parameter("tau0", self.tau0, TAU0_MIN, TAU0_MAX, sigma=None,
                                                      sigma_scale=TAU0_SCALE, p_shape=(), use="scipy")})

        if "sigma_init" in self.parameters:
            self.logger.info("...sigma_init...")
            parameters.update(
                {"sigma_init": generate_lognormal_parameter("sigma_init", self.sigma_init, 0.0, 10*self.sigma_init,
                                                            sigma=self.sigma_init, p_shape=(), use="scipy")})

        self.logger.info("...observation's model parameters...")
        if "epsilon" in self.parameters:
            self.logger.info("...epsilon...")
            parameters.update(
                {"epsilon": generate_lognormal_parameter("epsilon", self.epsilon, 0.0, 10*self.epsilon,
                                                         sigma=self.epsilon, p_shape=(), use="scipy")})
            
        if "scale" in self.parameters:
            self.logger.info("...scale...")
            parameters.update(
                {"scale": generate_lognormal_parameter("scale", self.scale, 0.1, 10*self.scale,
                                                       sigma=self.scale, p_shape=(), use="scipy")})
            
        if "offset" in self.parameters:
            self.logger.info("...offset...")
            parameters.update(
                {"offset":
                     generate_stochastic_parameter("offset", self.offset-10.0, self.offset+10.0, p_shape=(),
                                                   probability_distribution=ProbabilityDistributionTypes.NORMAL,
                                                   optimize_pdf=False, use="scipy",
                                                   **{"mu": self.offset, "sigma": 1.0})})
        return parameters

    def generate_model(self, target_data_type=TARGET_DATA_TYPE.SYNTHETIC.value, ground_truth={}):
        tic = time.time()
        self.logger.info("Generating model by " + self.__class__.__name__ + "...")
        parameters = self.generate_parameters()
        model = ODEStatisticalModel(self.name, self.number_of_regions, target_data_type, self.xmode, self.priors_mode,
                                    parameters, ground_truth, self.model_config, self.observation_model,
                                    self.sigma_x, self.sigma_init, self.tau1, self.tau0,
                                    self.scale, self.offset, self.epsilon,
                                    self.number_of_signals, self.time_length, self.dt, self.active_regions)
        self.logger.info(self.__class__.__name__  + " took " +
                         str(time.time() - tic) + ' sec for model generation')
        return model


class SDEStatisticalModelBuilder(ODEStatisticalModelBuilder):

    parameters = [XModes.X0MODE.value, "sigma_"+XModes.X0MODE.value, "tau1", "K", "x1init", "zinit",
                  "sigma_init", "dX1t", "dZt", "sigma", "epsilon", "scale", "offset"]
    sigma = SIGMA_DEF
    sde_mode = SDE_MODES.NONCENTERED.value

    def __init__(self, model_name="vep_sde", model_config=ModelConfiguration(),
                 parameters=[XModes.X0MODE.value, "sigma_"+XModes.X0MODE.value, "tau1", "K", "x1init", "zinit",
                             "sigma_init",  "dX1t", "dZt", "sigma", "epsilon", "scale", "offset"],
                 xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value,
                 sigma_x=None, sigma_x_scale=3, MC_direction_split=0.5,
                 sigma_init=SIGMA_INIT_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF, epsilon=EPSILON_DEF, sigma=SIGMA_DEF,
                 scale=SCALE_SIGNAL_DEF, offset=OFFSET_SIGNAL_DEF,
                 sde_mode=SDE_MODES.NONCENTERED.value, observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER.value,
                 number_of_signals=0, time_length=0, dt=DT_DEF, active_regions=[]):
        super(SDEStatisticalModelBuilder, self).__init__(model_name, model_config, parameters, xmode, priors_mode,
                                                         sigma_x, sigma_x_scale, MC_direction_split, sigma_init,
                                                         tau1, tau0, epsilon, scale, offset, observation_model,
                                                         number_of_signals, time_length, dt, active_regions)
        self.sigma_init = sigma_init
        self.sde_mode = sde_mode
        self.sigma = sigma

    def _repr(self, d=OrderedDict()):
        d.update(super(SDEStatisticalModelBuilder, self)._repr(d))
        nKeys = len(d)
        for ikey, (key, val) in enumerate(self.__dict__.iteritems()):
            d.update({str(nKeys+ikey) + ". " + key: str(val)})
        return d

    def generate_parameters(self):
        parameters = super(SDEStatisticalModelBuilder, self).generate_parameters()
        self.logger.info("Generating model parameters by " + self.__class__.__name__ + "...")
        if "sigma" in self.parameters:
            self.logger.info("...sigma...")
            parameters.update(
                {"sigma": generate_stochastic_parameter("sigma", 0.0, 10*self.sigma, p_shape=(),
                                                        probability_distribution=ProbabilityDistributionTypes.GAMMA,
                                                        optimize_pdf=True, use="scipy", **{"mean": 1.0, "skew": 0.0}).
                                              update_loc_scale(use="scipy", **{"mean": self.sigma, "std": self.sigma})})
        names = []
        mins = []
        maxs = []
        means = []
        if self.sde_mode == SDE_MODES.CENTERED.value:
            self.logger.info("...autoregression centered time series parameters...")
            if "x1" in self.parameters:
                names.append("x1")
                mins.append(X1_MIN)
                maxs.append(X1_MAX)
                means.append(X1_REST)
            if "z" in self.parameters:
                names.append("z")
                mins.append(Z_MIN)
                maxs.append(Z_MAX)
                means.append(calc_eq_z(X1_REST, self.model_config.yc, self.model_config.Iext1, "2d", x2=0.0,
                               slope=self.model_config.slope, a=self.model_config.a, b=self.model_config.b,
                               d=self.model_config.d, x1_neg=True))
            n_xp =len(names)
        else:
            self.logger.info("...autoregression noncentered time series parameters...")
            names = list(set(["dX1t", "dZt"]) & set(self.parameters))
            n_xp = len(names)
            mins = n_xp*[-1.0]
            maxs = n_xp*[1.0]
            means = n_xp*[0.0]
        for iV in range(n_xp):
            self.logger.info("..." + names[iV] + "...")
            parameters.update(
                {names[iV]: generate_stochastic_parameter(names[iV], mins[iV], maxs[iV],
                                                         p_shape=(self.time_length, self.number_of_regions),
                                                         probability_distribution=ProbabilityDistributionTypes.NORMAL,
                                                         optimize_pdf=False, use="scipy",
                                                         **{"mu": means[iV], "sigma": self.sigma})})
        return parameters

    def generate_model(self, target_data_type=TARGET_DATA_TYPE.SYNTHETIC.value, ground_truth={}):
        tic = time.time()
        self.logger.info("Generating model by " + self.__class__.__name__ + "...")
        parameters = self.generate_parameters()
        model = SDEStatisticalModel(self.name, self.number_of_regions, target_data_type, self.xmode, self.priors_mode,
                                    parameters, ground_truth, self.model_config, self.observation_model,
                                    self.sigma_x, self.sigma_init, self.sigma, self.tau1, self.tau0,
                                    self.scale, self.offset, self.epsilon,
                                    self.number_of_signals, self.time_length, self.dt, self.active_regions,
                                    self.sde_mode)
        self.logger.info(self.__class__.__name__  + " took " +
                         str(time.time() - tic) + ' sec for model generation')
        return model
