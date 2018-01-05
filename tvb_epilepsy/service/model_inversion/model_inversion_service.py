
import time
from copy import deepcopy

import numpy as np

from tvb_epilepsy.base.constants.model_constants import X1_EQ_CR_DEF, X1_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error, raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import copy_object_attributes, construct_import_path
from tvb_epilepsy.base.h5_model import convert_to_h5_model
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


LOG = initialize_logger(__name__)


STATISTICAL_MODEL_TYPES=["vep_sde"] #, "vep_ode", "vep_lsa"]


class ModelInversionService(object):

    X1EQ_MIN = -2.0
    X1_REST = X1_DEF
    X1_EQ_CR = X1_EQ_CR_DEF
    TAU1_DEF = 0.5
    TAU1_MIN = 0.1
    TAU1_MAX = 1.0
    TAU0_DEF = 30.0
    TAU0_MIN = 3.0
    TAU0_MAX = 3000.0
    K_MIN = 0.0
    K_MAX = 3.0
    MC_MIN = 0.0
    MC_MAX = 1.0
    MAX_MIN_RATIO = 1000.0

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None, model_name="",
                 logger=LOG, **kwargs):
        self.logger = logger
        self.model_name = model_name
        self.model_generation_time = 0.0
        self.target_data_type = ""
        self.observation_shape = ()
        for constant, default in zip(["X1EQ_MIN", "X1_REST", "X1_EQ_CR", "TAU1_DEF", "TAU1_MIN", "TAU1_MAX", "TAU0_DEF",
                                      "TAU0_MIN", "TAU0_MAX", "K_MIN", "K_MAX", "MC_MIN", "MC_MAX", "MAX_MIN_RATIO"],
                            [-2.0, X1_DEF, X1_EQ_CR_DEF, 0.5, 0.1, 1.0, 30.0, 3.0, 3000.0, 0.0, 3.0, 0.0, 1.0, 1000.0]):
            setattr(self, constant, kwargs.get(constant, default))
        if isinstance(model_configuration, ModelConfiguration):
            self.logger.info("Input model configuration set...")
            self.n_regions = model_configuration.n_regions
            self._copy_attributes(model_configuration,
                                  ["K", "x1EQ", "zEQ", "e_values", "x0_values"], deep_copy=True)
            self.model_connectivity = deepcopy(kwargs.pop("model_connectivity", model_configuration.model_connectivity))
            self.epileptor_parameters = self.get_epileptor_parameters(model_configuration)
        else:
            raise_value_error("Invalid input model configuration!:\n" + str(model_configuration))
        self.lsa_propagation_strengths = kwargs.pop("lsa_propagation_strengths", None)
        self.hypothesis_type = kwargs.pop("hypothesis_type", None)
        if isinstance(hypothesis, DiseaseHypothesis):
            self._copy_attributes(hypothesis, ["lsa_propagation_strengths", "type"],
                                  ["lsa_propagation_strengths", "hypothesis_type"],  deep_copy=True, check_none=True)
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
            if not(isinstance(sensors, list)):
                sensors = head.get_sensors_id(sensor_ids=kwargs.pop("seeg_sensor_id", 0), s_type=Sensors.TYPE_SEEG)
        if isinstance(connectivity, Connectivity):
            self._copy_attributes(connectivity, ["region_labels", "centres", "orientations"],
                                  ["region_labels", "region_centers", "region_orientations"], deep_copy=True, check_none=True)
        if isinstance(sensors, Sensors):
            self._copy_attributes(sensors, ["labels", "locations", "gain_matrix"],
                                  ["sensors_labels", "sensors_locations", "gain_matrix"], deep_copy=True, check_none=True)
        self.dynamical_model = dynamical_model
        self.MC_direction_split = kwargs.get("MC_direction_split", 0.5)
        self.default_parameters = {}
        self.__set_default_parameters(**kwargs)
        self.logger.info("Model Inversion Service instance created!")
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.context_str += "; from tvb_epilepsy.base.model.model_configuration import ModelConfiguration"
        self.create_str = "ModelInversionService(ModelConfiguration())"

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "ModelInversionService")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

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
        if np.in1d(self.dynamical_model, AVAILABLE_DYNAMICAL_MODELS_NAMES):
            tau1_def = EPILEPTOR_MODEL_TAU1[self.dynamical_model]
            tau0_def = EPILEPTOR_MODEL_TAU0[self.dynamical_model]
        else:
            tau1_def = self.TAU1_DEF
            tau0_def = self.TAU0_DEF
        return tau1_def, tau0_def

    def get_default_K(self):
        return np.maximum(np.mean(self.K) / np.mean(self.MC_direction_split), 0.001)

    def get_default_MC(self):
        MC_def = self.get_SC()
        inds = np.triu_indices(self.n_regions, 1)
        MC_def[inds] = MC_def[inds] * self.MC_direction_split
        inds = np.tril_indices(self.n_regions, 1)
        MC_def[inds] = MC_def[inds] * (1.0 - self.MC_direction_split)
        return MC_def

    def get_SC(self):
        # Set symmetric connectivity to be in the interval [self.MC_MAX / self.MAX_MIN_RATIO, self.MC_MAX],
        # where self.MC_MAX corresponds to the 95th percentile of model_connectivity
        p95 = np.percentile(self.model_connectivity.flatten(), 95)
        SC = np.array(self.model_connectivity)
        if p95 != self.MC_MAX:
            SC = SC / p95
            SC[SC > self.MC_MAX] = 1.0
        mc_def_min = self.MC_MAX / self.MAX_MIN_RATIO
        SC[SC < mc_def_min] = mc_def_min
        diag_ind = range(self.n_regions)
        SC[diag_ind, diag_ind] = 0.0
        return SC

    def get_default_sig_eq(self, **kwargs):
        return kwargs.get("sig_eq", (self.X1_EQ_CR - self.X1_REST) / 10.0)

    def __set_default_parameters(self, **kwargs):
        # Generative model:
        # Epileptor:
        self.default_parameters.update(set_parameter_defaults("x1eq", "normal", (self.n_regions,),
                                                              self.X1EQ_MIN, X1_EQ_CR_DEF,
                                                              pdf_params={"mean": np.maximum(self.x1EQ, self.X1_REST),
                                                                          "sigma": (self.X1_CR - self.X1_REST) / 10}))
        self.default_parameters.update(set_parameter_defaults("K", "lognormal", (),
                                                              self.K_MIN,  self.K_MAX,
                                                              self.get_default_K(),
                                                              lambda m: m/kwargs.get("K_scale_zscore", 6.0),
                                                              **kwargs))
        tau1, tau0 = self.get_default_taus()
        self.default_parameters.update(set_parameter_defaults("tau1", "lognormal", (),               # name, pdf, shape
                                                              self.TAU1_MIN, self.TAU1_MAX,          # min, max
                                                              tau1,                                 # mean
                                                              lambda m: m / kwargs.get("tau1_scale_zscore", 6.0), # std
                                                              pdf_params={"std": 1.0, "skew": 0.0},
                                                              **kwargs))
        self.default_parameters.update(set_parameter_defaults("tau0", "lognormal", (),
                                                              self.TAU0_MIN, self.TAU0_MAX,
                                                              tau0,
                                                              lambda m: m / kwargs.get("tau0_scale_zscore", 6.0), # std,
                                                              **kwargs))
        # Coupling:
        MC_split_std = np.min([self.MC_direction_split, 1.0 - self.MC_direction_split]) \
                       / kwargs.get("MC_split_scale_zscore", 6.0)
        self.default_parameters.update(set_parameter_defaults("MC_split", "normal",
                                                              (self.n_regions * (self.n_regions-1)/2,),
                                                              0.0, 1.0,
                                                              self.MC_direction_split, MC_split_std,
                                                              **kwargs))
        MC_def = self.get_default_MC()
        self.default_parameters.update(set_parameter_defaults("MC", "normal", (self.n_regions, self.n_regions),
                                                              0.0, self.MC_MAX,
                                                              MC_def,                                     # mean
                                                              lambda m: m / kwargs.get("MC_scale_zscore", 6.0),  # std
                                                              **kwargs))
        # Integration:
        # sig_eq_def = self.get_default_sig_eq(**kwargs)
        # self.default_parameters.update(set_parameter_defaults("sig_eq", "lognormal", (),
        #                                                       0.0, 3 * sig_eq_def,
        #                                                       sig_eq_def, sig_eq_def / 3.0, **kwargs))
        # Observation model
        self.default_parameters.update(set_parameter_defaults("eps", "lognormal", (),
                                                              0.0, 0.5,
                                                              0.1, 0.1, **kwargs))

    def generate_statistical_model(self, model_name=None, **kwargs):
        if model_name is None:
            model_name = self.model_name
        tic = time.time()
        self.logger.info("Generating model...")
        self.default_parameters.update(kwargs)
        model = StatisticalModel(model_name, self.n_regions, self.get_default_sig_eq(**kwargs),
                                 **self.default_parameters)
        self.model_generation_time = time.time() - tic
        self.logger.info(str(self.model_generation_time) + ' sec required for model generation')
        return model


