
import time
from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.constants.model_constants import X1_EQ_CR_DEF, X1_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error, raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import copy_object_attributes, construct_import_path
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r
from tvb_epilepsy.base.model.vep.connectivity import Connectivity
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import generate_stochastic_parameter
from tvb_epilepsy.base.model.statistical_models.statistical_model import StatisticalModel
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
        self.lsa_propagation_strengths = kwargs.get("lsa_propagation_strengths", None)
        self.hypothesis_type = kwargs.get("hypothesis_type", None)
        if isinstance(hypothesis, DiseaseHypothesis):
            self._copy_attributes(hypothesis, ["lsa_propagation_strengths", "type"],
                                  ["lsa_propagation_strengths", "hypothesis_type"],  deep_copy=True, check_none=True)
            self.logger.info("Input hypothesis set...")
        self.projection = kwargs.get("projection", None)
        self.sensors_locations = kwargs.get("sensors_locations", None)
        self.sensors_labels = kwargs.get("sensors_labels", None)
        sensors = kwargs.get("sensors", None)
        self.region_centers = kwargs.get("region_centers", None)
        self.region_labels = kwargs.get("region_labels", None)
        self.region_orientations = kwargs.get("region_orientations", None)
        connectivity = kwargs.get("connectivity", None)
        if isinstance(head, Head):
            connectivity = head.connectivity
            if not(isinstance(sensors, Sensors)):
                sensors = head.get_sensors_id(sensor_ids=kwargs.get("seeg_sensor_id", 0),
                                              sensors_type=Sensors.TYPE_SEEG)
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
        self.sig_eq = self.get_default_sig_eq(x1eq_def=kwargs.get("x1eq_def", X1_DEF),
                                              x1eq_cr=kwargs.get("x1eq_cr", X1_EQ_CR_DEF))
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

    def generate_model_parameters(self, **kwargs):
        parameters = OrderedDict()
        # Generative model:
        # Epileptor:
        parameter = kwargs.get("x1eq", None)
        if not(isinstance(parameter, Parameter)):
            x1eq = np.maximum(kwargs.get("x1eq", self.x1EQ), X1_DEF)
            parameter = generate_stochastic_parameter("x1eq",
                                                      low=kwargs.get("x1eq_lo", self.X1EQ_MIN),
                                                      high=kwargs.get("x1eq_hi", X1_EQ_CR_DEF),
                                                      p_shape=(self.n_regions,),
                                                      probability_distribution="normal",
                                                      optimize=False,
                                                      mean=x1eq, sigma=kwargs.get("x1eq_sig", 0.1))
        parameters.update({parameter.name: parameter})

        parameter = kwargs.get("K", None)
        if not(isinstance(parameter, Parameter)):
            K_def = np.maximum(kwargs.get("K_def", np.mean(self.K)), 0.1)
            pdf_params = kwargs.get("K_pdf_params", {"mean": K_def, "std": kwargs.get("K_sig", K_def)})
            parameter = generate_stochastic_parameter("K",
                                                      low=kwargs.get("K_lo", self.K_MIN),
                                                      high=kwargs.get("K_hi", self.K_MAX),  p_shape=(),
                                                      probability_distribution= kwargs.get("K_pdf", "lognormal"),
                                                      optimize=True, **pdf_params)
        parameters.update({parameter.name: parameter})

        # tau1_def = kwargs.get("tau1_def", 0.5)
        parameter = kwargs.get("tau1", None)
        if not (isinstance(parameter, Parameter)):
            tau1_def = kwargs.get("tau1_def", self.tau1)
            pdf_params = kwargs.get("tau1_pdf_params", {"mean": tau1_def, "std": kwargs.get("tau1_sig", tau1_def)})
            parameter = generate_stochastic_parameter("tau1",
                                                      low=kwargs.get("tau1_lo", self.TAU1_MIN),
                                                      high=kwargs.get("tau1_hi", self.TAU1_MAX),
                                                      p_shape=(),
                                                      probability_distribution=kwargs.get("tau1", "lognormal"),
                                                      optimize=True, **pdf_params)
        parameters.update({parameter.name: parameter})

        parameter = kwargs.get("tau0", None)
        if not(isinstance(parameter, Parameter)):
            tau0_def = kwargs.get("tau0_def", self.tau0)
            pdf_params = kwargs.get("tau0_pdf_params", {"mean": tau0_def, "std": kwargs.get("tau0_sig", tau0_def)})
            parameter = generate_stochastic_parameter("tau0",
                                                      low=kwargs.get("tau0_lo", self.TAU0_MIN),
                                                      high=kwargs.get("tau0_hi", self.TAU0_MAX),
                                                      p_shape=(),
                                                      probability_distribution=kwargs.get("tau0_pdf", "lognormal"),
                                                      optimize=True, **pdf_params)
        parameters.update({parameter.name: parameter})

        # Coupling:
        parameter = kwargs.get("MC", None)
        if not(isinstance(parameter, Parameter)):
            model_connectivity = kwargs.get("structural_connectivity", self.model_connectiviy)
            p0595 = np.percentile(model_connectivity.flatten(), [5, 95])
            mean = np.maximum(p0595[0], model_connectivity)
            pdf_params = kwargs.get("MC_pdf_params", {"mean": mean, "std": kwargs.get("MC_sig", mean/3.0)})
            parameter = generate_stochastic_parameter("MC",
                                                      low=kwargs.get("MC_lo", self.MC_MIN),
                                                      high=kwargs.get("MC_hi", 3 * p0595[1]),
                                                      p_shape=(self.n_regions, self.n_regions),
                                                      probability_distribution=kwargs.get("MC_pdf", "lognormal"),
                                                      optimize=True, **pdf_params)
        parameters.update({parameter.name: parameter})

        # Integration:
        parameter = kwargs.get("sig_eq", None)
        if not(isinstance(parameter, Parameter)):
            sig_eq_def = kwargs.get("sig_eq_def", 0.1)
            pdf_params = kwargs.get("sig_eq_pdf_params", {"mean": sig_eq_def, 
                                                          "std": kwargs.get("sig_eq_sig", sig_eq_def)})
            parameter = generate_stochastic_parameter("sig_eq",
                                                      low=kwargs.get("sig_eq_lo", 0.0),
                                                      high=kwargs.get("sig_eq_hi", 2 * sig_eq_def),
                                                      p_shape=(),
                                                      probability_distribution=kwargs.get("sig_eq_pdf", "lognormal"),
                                                      optimize=True, **pdf_params)
        parameters.update({parameter.name: parameter})

        # Observation model
        parameter = kwargs.get("eps", None)
        if not(isinstance(parameter, Parameter)):
            eps_def = kwargs.get("eps_def", 0.1)
            pdf_params = kwargs.get("eps_pdf_params", {"mean": eps_def, "std": kwargs.get("eps_sig", eps_def)})
            parameter = generate_stochastic_parameter("eps",
                                                      low=kwargs.get("eps_lo", 0.0),
                                                      high=kwargs.get("eps_hi", 1.0),
                                                      probability_distribution=kwargs.get("eps_pdf", "lognormal"),
                                                      optimize=True, **pdf_params)
        parameters.update({parameter.name: parameter})
        return parameters
        
    def generate_statistical_model(self, model_name=None, **kwargs):
        if model_name is None:
            model_name = self.model_name
        tic = time.time()
        self.logger.info("Generating model...")
        model = StatisticalModel(model_name, self.generate_model_parameters( **kwargs), self.n_regions)
        self.model_generation_time = time.time() - tic
        self.logger.info(str(self.model_generation_time) + ' sec required for model generation')
        return model


