import numpy as np
from tvb.simulator.models import Epileptor

from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r
from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, X1_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.vep.connectivity import Connectivity
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.custom.simulator_custom import EpileptorModel
from tvb_epilepsy.tvb_api.epileptor_models import *

AVAILABLE_DYNAMICAL_MODELS = (Epileptor, EpileptorModel, EpileptorDP2D, EpileptorDP, EpileptorDPrealistic)

LOG = initialize_logger(__name__)


class ModelInversionService(object):

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None,
                 model=None, model_code=None, model_code_path="", target_data=None, target_data_type="",
                 logger=LOG, **kwargs):
        self.results = {}
        self.model = model
        self.model_code = model_code
        self.model_code_path = model_code_path
        self.target_data_type = target_data_type
        self.target_data = target_data
        if self.target_data is not None:
            self.observation_shape = target_data.shape
        else:
            self.observation_shape = 0
        if isinstance(model_configuration, ModelConfiguration):
            self.model_config = model_configuration
            logger.info("Input model configuration set...")
        else:
            raise_value_error("Invalid input model configuration!:\n" + str(model_configuration))
        if isinstance(hypothesis, DiseaseHypothesis):
            self.hypothesis = hypothesis
            logger.info("Input hypothesis set...")
        if isinstance(head, Head):
            self.head = head
            logger.info("Input head set...")
        if isinstance(dynamical_model, AVAILABLE_DYNAMICAL_MODELS):
            self.dynamical_model = dynamical_model
        logger.info("Model Inversion Service instance created!")

    def get_epileptor_parameters(self, logger=LOG):
        logger.info("Unpacking epileptor parameters...")
        epileptor_params = {}
        for p in ["a", "b", "d", "yc", "Iext1", "slope"]:
            temp = getattr(self.model_config, p)
            if isinstance(temp, (np.ndarray, list)):
                if np.all(temp[0], np.array(temp)):
                    temp = temp[0]
                else:
                    raise_not_implemented_error("Statistical models where not all regions have the same value " +
                                                " for parameter " + p + " are not implemented yet!")
            self.epileptor_params.update({p: temp})
        x0cr, rx0 = calc_x0cr_r(epileptor_params["yc"], epileptor_params["Iext1"], epileptor_params["a"],
                                epileptor_params["b"], epileptor_params["d"], zmode=np.array("lin"),
                                x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF, test=False,
                                shape=None, calc_mode="non_symbol")
        epileptor_params.update({"x0cr": x0cr, "rx0": rx0})
        return epileptor_params

    def get_default_tau0(self):
        if isinstance(self.dynamical_model, AVAILABLE_DYNAMICAL_MODELS):
            if isinstance(self.dynamic_model, (Epileptor, EpileptorModel)):
                return np.mean(self.dynamical_model.tt)
            elif isinstance(self.dynamic_model, (EpileptorDP, EpileptorDP2D, EpileptorDPrealistic)):
                return np.mean(self.dynamical_model.tau0)
        else:
            return 30.0

    def get_default_tau1(self):
        if isinstance(self.dynamical_model, AVAILABLE_DYNAMICAL_MODELS):
            if isinstance(self.dynamic_model, (Epileptor, EpileptorModel)):
                return np.mean(1.0 / self.dynamical_model.r)
            elif isinstance(self.dynamic_model, (EpileptorDP, EpileptorDP2D, EpileptorDPrealistic)):
                return np.mean(self.dynamical_model.tau1)
        else:
            return 0.5

    def get_region_labels(self, raise_error=False):
        region_labels = None
        if self.head is not None:
            if isinstance(self.head.connectivity, Connectivity):
                region_labels = self.head.connectivity.region_labels
        if region_labels is None and raise_error:
            raise_value_error("No region labels found!")
        else:
            return region_labels

    def get_default_sig_eq(self, x1eq_def=X1_DEF, x1eq_cr=X1_EQ_CR_DEF):
        return (x1eq_cr - x1eq_def) / 3.0