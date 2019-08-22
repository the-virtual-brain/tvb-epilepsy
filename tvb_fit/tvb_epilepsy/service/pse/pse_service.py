from copy import deepcopy

import numpy as np

from tvb_fit.service.pse.pse_service import ABCPSEService
from tvb_fit.tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_fit.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder

from tvb_scripts.utils.log_error_utils import initialize_logger


class PSEService(ABCPSEService):

    logger = initialize_logger(__name__)

    def update_model_config(self, params, conn_matrix=None, model_config_builder_input=None, hypothesis=None,
                            x1eq_mode="optimize"):
        # Create a ModelConfigService and update it
        if isinstance(model_config_builder_input, ModelConfigurationBuilder):
            model_configuration_builder = deepcopy(model_config_builder_input)
            if isinstance(conn_matrix, np.ndarray):
                model_configuration_builder.connectivity = conn_matrix
        else:
            model_configuration_builder = ModelConfigurationBuilder(connectivity=conn_matrix, x1eq_mode=x1eq_mode)
        model_configuration_builder.set_attributes_from_pse(params, self.params_paths, self.params_indices)
        # Copy and update hypothesis
        if isinstance(hypothesis, DiseaseHypothesis):
            hypo_copy = deepcopy(hypothesis)
            hypo_copy.update_for_pse(params, self.params_paths, self.params_indices)
        else:
            hypo_copy = DiseaseHypothesis(model_configuration_builder.number_of_regions)
        # Obtain ModelConfiguration
        if hypothesis.type == "Epileptogenicity":
            model_configuration = model_configuration_builder.build_model_from_E_hypothesis(hypo_copy)
        else:
            model_configuration = model_configuration_builder.build_model_from_hypothesis(hypo_copy)
        return model_configuration, hypo_copy

