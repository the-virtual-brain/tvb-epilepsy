import time
from copy import deepcopy
import numpy as np
from tvb_epilepsy.base.constants.model_constants import X1_EQ_CR_DEF, X1_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.constants.model_inversion_constants import X1EQ_MIN, X1EQ_MAX, MC_SCALE, \
                     TAU1_DEF, TAU1_MIN, TAU1_MAX, TAU0_DEF, TAU0_MIN, TAU0_MAX, K_MIN, K_MAX, MC_MAX, MC_MAX_MIN_RATIO
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error, raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import copy_object_attributes
from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r
from tvb_epilepsy.base.model.vep.connectivity import Connectivity
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.statistical_models.statistical_model import StatisticalModel
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter_defaults
from tvb_epilepsy.service.epileptor_model_factory import AVAILABLE_DYNAMICAL_MODELS_NAMES, EPILEPTOR_MODEL_TAU1, \
    EPILEPTOR_MODEL_TAU0


STATISTICAL_MODEL_TYPES=["vep_sde"] #, "vep_ode", "vep_lsa"]


class ModelInversionService(object):
    logger = initialize_logger(__name__)

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None, model_name="", **kwargs):

        self.model_name = model_name
        self.model_generation_time = 0.0
        self.target_data_type = ""
        self.observation_shape = ()
        if isinstance(model_configuration, ModelConfiguration):
            self.logger.info("Input model configuration set...")
            self.n_regions = model_configuration.n_regions
            self._copy_attributes(model_configuration,
                                  ["K", "x1EQ", "zEQ", "e_values", "x0_values", "x0"], deep_copy=True)
            self.model_connectivity = deepcopy(kwargs.pop("model_connectivity", model_configuration.model_connectivity))
            self.epileptor_parameters = self.get_epileptor_parameters(model_configuration)
        else:
            raise_value_error("Invalid input model configuration!:\n" + str(model_configuration))
        self.lsa_propagation_strengths = kwargs.pop("lsa_propagation_strengths", None)
        self.hypothesis_type = kwargs.pop("hypothesis_type", None)
        if isinstance(hypothesis, DiseaseHypothesis):
            self._copy_attributes(hypothesis, ["lsa_propagation_strengths", "type"],
                                  ["lsa_propagation_strengths", "hypothesis_type"], deep_copy=True, check_none=True)
            self.logger.info("Input hypothesis set...")
        self.gain_matrix = kwargs.pop("gain_matrix", None)
        self.sensors_locations = kwargs.pop("sensors_locations", None)
        self.sensors_labels = kwargs.pop("sensors_labels", None)
        sensors = kwargs.pop("sensors", None)
        self.region_centers = kwargs.pop("region_centers", None)
        self.region_labels = kwargs.pop("region_labels", None)
        self.region_orientations = kwargs.pop("region_orientations", None)
        connectivity = kwargs.pop("connectivity", None)
        if isinstance(head, Head):
            connectivity = head.connectivity
            if not (isinstance(sensors, list)):
                sensors = head.get_sensors_id(sensor_ids=kwargs.pop("seeg_sensor_id", 0), s_type=Sensors.TYPE_SEEG)
        if isinstance(connectivity, Connectivity):
            self._copy_attributes(connectivity, ["region_labels", "centres", "orientations"],
                                  ["region_labels", "region_centers", "region_orientations"], deep_copy=True,
                                  check_none=True)
        if isinstance(sensors, Sensors):
            self._copy_attributes(sensors, ["labels", "locations", "gain_matrix"],
                                  ["sensors_labels", "sensors_locations", "gain_matrix"], deep_copy=True, check_none=True)
        self.dynamical_model = dynamical_model
        self.MC_direction_split = kwargs.get("MC_direction_split", 0.5)
        self.default_parameters = {}
        self.__set_default_parameters(**kwargs)
        self.logger.info("Model Inversion Service instance created!")

    def _copy_attributes(self, obj, attributes_obj, attributes_self=None, deep_copy=False, check_none=False):
        copy_object_attributes(obj, self, attributes_obj, attributes_self, deep_copy, check_none)

    def get_epileptor_parameters(self, model_config):
        self.logger.info("Unpacking epileptor parameters...")
        epileptor_params = {}
        for p in ["a", "b", "d", "yc", "Iext1", "slope"]:
            temp = getattr(model_config, p)
            if isinstance(temp, (np.ndarray, list)):
                if np.all(temp[0], np.array(temp)):
                    temp = temp[0]
                else:
                    raise_not_implemented_error("Statistical models where not all regions have the same value " +
                                                " for parameter " + p + " are not implemented yet!")
            epileptor_params.update({p: temp})
        x0cr, rx0 = calc_x0cr_r(epileptor_params["yc"], epileptor_params["Iext1"], epileptor_params["a"],
                                epileptor_params["b"], epileptor_params["d"], zmode=np.array("lin"),
                                x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF, test=False,
                                shape=None, calc_mode="non_symbol")
        epileptor_params.update({"x0cr": x0cr, "rx0": rx0})
        return epileptor_params

    def get_default_taus(self):
        # if np.in1d(self.dynamical_model, AVAILABLE_DYNAMICAL_MODELS_NAMES):
        #     tau1_def = EPILEPTOR_MODEL_TAU1[self.dynamical_model]
        #     tau0_def = EPILEPTOR_MODEL_TAU0[self.dynamical_model]
        # else:
        tau1_def = TAU1_DEF
        tau0_def = TAU0_DEF
        return tau1_def, tau0_def

    def get_default_K(self):
        return np.maximum(np.mean(self.K) / np.mean(self.MC_direction_split), 0.001)

    def get_default_MC(self):
        MC_def = self.get_SC()
        inds = np.triu_indices(self.n_regions, 1)
        MC_def[inds] = MC_def[inds] * self.MC_direction_split
        inds = np.tril_indices(self.n_regions, 1)
        MC_def[inds] = MC_def[inds] * (1.0 - self.MC_direction_split)
        MC_def[MC_def < 0.001] = 0.001
        return MC_def

    def get_SC(self):
        # Set symmetric connectivity to be in the interval [MC_MAX / MAX_MIN_RATIO, MC_MAX],
        # where self.MC_MAX corresponds to the 95th percentile of model_connectivity
        p95 = np.percentile(self.model_connectivity.flatten(), 95)
        SC = np.array(self.model_connectivity)
        if p95 != MC_MAX:
            SC = SC / p95
            SC[SC > MC_MAX] = 1.0
        mc_def_min = MC_MAX / MC_MAX_MIN_RATIO
        SC[SC < mc_def_min] = mc_def_min
        diag_ind = range(self.n_regions)
        SC[diag_ind, diag_ind] = 0.0
        return SC

    def __set_default_parameters(self, **kwargs):
        # Generative model:
        # Epileptor:
        x1eq_max = kwargs.get("x1eq_max", X1EQ_MAX)
        x1eq_star_max = x1eq_max - X1EQ_MIN
        x1eq_star_mean = x1eq_max - self.x1EQ
        x1eq_std = np.minimum(kwargs.get("sig_eq", x1eq_star_max / 4.0), np.abs(x1eq_star_mean)/3.0)
        self.default_parameters.update(set_parameter_defaults("x1eq_star", "lognormal", (self.n_regions,),
                                                              0.0, x1eq_star_max,
                                                              x1eq_star_mean, x1eq_std,
                                                              pdf_params={"mean": x1eq_star_mean/x1eq_std,
                                                                          "skew": 0.0}, **kwargs))
        K_mean = self.get_default_K()
        K_std = np.min([K_mean-K_MIN, K_MAX-K_mean]) / kwargs.get("K_scale", 6.0)
        self.default_parameters.update(set_parameter_defaults("K", "lognormal", (),
                                                              K_MIN,  K_MAX,
                                                              K_mean, K_std,
                                                              pdf_params={"mean": K_mean/K_std, "skew": 0.0}, **kwargs))
        tau1_mean, tau0_mean = self.get_default_taus()
        tau1_std = np.min([tau1_mean - TAU1_MIN, TAU1_MAX - tau1_mean]) / kwargs.get("tau1_scale", 6.0)
        self.default_parameters.update(set_parameter_defaults("tau1", "lognormal", (),               # name, pdf, shape
                                                              TAU1_MIN, TAU1_MAX,          # min, max
                                                              tau1_mean, tau1_std,
                                                              pdf_params={"mean": tau1_mean/tau1_std, "skew": 0.0},
                                                              **kwargs))
        tau0_std = np.min([tau0_mean - TAU0_MIN, TAU0_MAX - tau0_mean]) / kwargs.get("tau0_scale", 6.0)
        self.default_parameters.update(set_parameter_defaults("tau0", "lognormal", (),
                                                              TAU0_MIN, TAU0_MAX,
                                                              tau0_mean, tau0_std,
                                                              pdf_params={"mean": tau0_mean/tau0_std, "skew": 0.0},
                                                              **kwargs))
        # Coupling:
        MCsplit_std = np.min([self.MC_direction_split, 1.0 - self.MC_direction_split]) \
                       / kwargs.get("MCsplit_scale", 10.0)
        self.default_parameters.update(set_parameter_defaults("MCsplit", "normal", # or "beta"...
                                                              (self.n_regions * (self.n_regions-1)/2,),
                                                              0.0, 1.0,
                                                              pdf_params={"mu": self.MC_direction_split,
                                                                          "sigma": MCsplit_std}, **kwargs))
        MC_def = self.get_default_MC()
        self.default_parameters.update(set_parameter_defaults("MC", "normal", (self.n_regions, self.n_regions),
                                                              0.0, MC_MAX,
                                                              pdf_params=
                                                              {"mu": MC_def,
                                                               "sigma": np.minimum(MC_def, MC_MAX - MC_def) /
                                                                        kwargs.get("MC_scale", 6.0)}, **kwargs))
        # Observation model
        self.default_parameters.update(set_parameter_defaults("eps", "lognormal", (),
                                                              0.0, 0.5,
                                                              0.1, 0.1 / kwargs.get("eps_scale", 3.0),
                                                              pdf_params={"mean": 1.0, "skew": 0.0}, **kwargs))

    def generate_statistical_model(self, model_name=None, **kwargs):
        if model_name is None:
            model_name = self.model_name
        tic = time.time()
        self.logger.info("Generating model...")
        self.default_parameters.update(kwargs)
        model = StatisticalModel(model_name, self.n_regions, kwargs.get("x1eq_min", X1EQ_MIN),
                                 kwargs.get("x1eq_max", X1EQ_MAX), kwargs.get("MC_scale", MC_SCALE),
                                 **self.default_parameters)
        self.model_generation_time = time.time() - tic
        self.logger.info(str(self.model_generation_time) + ' sec required for model generation')
        return model
