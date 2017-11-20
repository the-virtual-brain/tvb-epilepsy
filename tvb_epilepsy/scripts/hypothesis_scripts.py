
from tvb_epilepsy.base.constants.module_constants import EIGENVECTORS_NUMBER_SELECTION, WEIGHTED_EIGENVECTOR_SUM
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.service.lsa_service import LSAService


def start_lsa_run(hypothesis, connectivity_matrix, logger=None):

    if logger is None:
        logger = initialize_logger(__name__)

    logger.info("creating model configuration...")
    model_configuration_service = ModelConfigurationService(hypothesis.number_of_regions)
    model_configuration = model_configuration_service. \
        configure_model_from_hypothesis(hypothesis, connectivity_matrix)

    logger.info("running LSA...")
    lsa_service = LSAService(eigen_vectors_number_selection=EIGENVECTORS_NUMBER_SELECTION, eigen_vectors_number=None,
                             weighted_eigenvector_sum=WEIGHTED_EIGENVECTOR_SUM, normalize_propagation_strength=False)
    lsa_hypothesis = lsa_service.run_lsa(hypothesis, model_configuration)

    return model_configuration_service, model_configuration, lsa_service, lsa_hypothesis