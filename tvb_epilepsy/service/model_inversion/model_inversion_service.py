
import time

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


STATISTICAL_MODEL_TYPES=["vep_sde", "vep_dWt", "vep_ode", "vep_lsa"]


class ModelInversionService(object):

    X1EQ_MIN = -2.0
    TAU1_DEF = 0.5
    TAU1_MIN = 0.1
    TAU1_MAX = 1.0
    TAU0_DEF = 30.0
    TAU0_MIN = 3.0
    TAU0_MAX = 3000.0
    K_MIN = 0.0
    K_MAX = 2.0
    MC_MIN = 0.0

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None, model_name="",
                 logger=LOG, **kwargs):
        self.logger = logger
        self.model_name = model_name
        self.model_generation_time = 0.0
        self.target_data_type = ""
        self.observation_shape = ()
        if isinstance(model_configuration, ModelConfiguration):
            self.logger.info("Input model configuration set...")
            self.n_regions = model_configuration.n_regions
            self._copy_attributes(model_configuration,
                                  ["K", "x1EQ", "zEQ", "e_values", "x0_values", "model_connectivity"], deep_copy=True)
            self.epileptor_parameters = self.get_epileptor_parameters(model_configuration)
        else:
            raise_value_error("Invalid input model configuration!:\n" + str(model_configuration))
        self.lsa_propagation_strengths = kwargs.pop("lsa_propagation_strengths", None)
        self.hypothesis_type = kwargs.pop("hypothesis_type", None)
        if isinstance(hypothesis, DiseaseHypothesis):
            self._copy_attributes(hypothesis, ["lsa_propagation_strengths", "type"],
                                  ["lsa_propagation_strengths", "hypothesis_type"],  deep_copy=True, check_none=True)
            self.logger.info("Input hypothesis set...")
        self.projection = kwargs.pop("projection", None)
        self.sensors_locations = kwargs.pop("sensors_locations", None)
        self.sensors_labels = kwargs.pop("sensors_labels", None)
        sensors = kwargs.pop("sensors", None)
        self.region_centers = kwargs.pop("region_centers", None)
        self.region_labels = kwargs.pop("region_labels", None)
        self.region_orientations = kwargs.pop("region_orientations", None)
        connectivity = kwargs.pop("connectivity", None)
        if isinstance(head, Head):
            connectivity = head.connectivity
            if not(isinstance(sensors, Sensors)):
                sensors = head.get_sensors_id(sensor_ids=kwargs.pop("seeg_sensor_id", 0), s_type=Sensors.TYPE_SEEG)
        if isinstance(connectivity, Connectivity):
            self._copy_attributes(connectivity, ["region_labels", "centers", "orientations"],
                                  ["region_labels", "region_centers", "region_orientations"], deep_copy=True, check_none=True)
        if isinstance(sensors, Sensors):
            self._copy_attributes(sensors, ["labels", "locations", "projection"],
                                  ["sensors_labels", "sensors_locations", "projection"], deep_copy=True, check_none=True)
        self.tau1 = self.TAU1_DEF
        self.tau0 = self.TAU0_DEF
        if np.in1d(dynamical_model, AVAILABLE_DYNAMICAL_MODELS_NAMES):
            self.tau1 = self.get_default_tau1(dynamical_model)
            self.tau0 = self.get_default_tau0(dynamical_model)
        self.sig_eq = self.get_default_sig_eq(x1eq_def=kwargs.pop("x1eq_def", X1_DEF),
                                              x1eq_cr=kwargs.pop("x1eq_cr", X1_EQ_CR_DEF))
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

    def get_default_tau1(self, dynamical_model):
        return EPILEPTOR_MODEL_TAU1[dynamical_model]

    def get_default_tau0(self, dynamical_model):
        return EPILEPTOR_MODEL_TAU0[dynamical_model]

    def get_default_sig_eq(self, x1eq_def=X1_DEF, x1eq_cr=X1_EQ_CR_DEF):
        return (x1eq_cr - x1eq_def) / 3.0

    def __set_default_parameters(self, **kwargs):
        # Generative model:
        # Epileptor:
        self.default_parameters.update(set_parameter_defaults("x1eq", "normal", (self.n_regions,),
                                                              self.X1EQ_MIN, X1_EQ_CR_DEF,
                                                              np.maximum(self.x1EQ, X1_DEF), sigma=0.03))
        self.default_parameters.update(set_parameter_defaults("K", "lognormal", (),         # name, pdf, shape
                                                              self.K_MIN,  self.K_MAX,      # min, max
                                                              np.mean(self.K), lambda m: m/6, **kwargs))   # mean, (std)
        self.default_parameters.update(set_parameter_defaults("tau1", "lognormal", (),
                                                              self.TAU1_MIN, self.TAU1_MAX,
                                                              self.tau1, lambda m: m/3,  **kwargs))
        self.default_parameters.update(set_parameter_defaults("tau0", "lognormal", (),
                                                              self.TAU0_MIN, self.TAU0_MAX,
                                                              self.tau0, lambda m: m/3, **kwargs))
        # Coupling:
        model_connectivity = kwargs.get("model_connectivity", self.model_connectivity)
        p0595 = np.percentile(model_connectivity.flatten(), [5, 95])
        mean = np.minimum(np.maximum(np.maximum(p0595[0], 0.001), model_connectivity), p0595[1])
        self.default_parameters.update(set_parameter_defaults("MC", "lognormal", (),
                                                              self.MC_MIN, 3 * p0595[1],
                                                              mean, lambda m: m/6.0, **kwargs))
        # Integration:
        self.default_parameters.update(set_parameter_defaults("sig_eq", "lognormal", (),
                                                              0.0, lambda s: 3 * s,
                                                              0.03, lambda m: m / 6.0, **kwargs))
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
        model = StatisticalModel(model_name, self.n_regions, **self.default_parameters)
        self.model_generation_time = time.time() - tic
        self.logger.info(str(self.model_generation_time) + ' sec required for model generation')
        return model


