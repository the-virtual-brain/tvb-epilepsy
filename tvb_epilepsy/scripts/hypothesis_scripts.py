
import os

import numpy as np

from tvb_epilepsy.base.constants.configurations import DATA_CUSTOM, FOLDER_RES, FOLDER_FIGURES
from tvb_epilepsy.base.constants.module_constants import TVB, DATA_MODE, EIGENVECTORS_NUMBER_SELECTION, \
                                                                                                WEIGHTED_EIGENVECTOR_SUM
from tvb_epilepsy.base.constants.model_constants import X0_DEF, E_DEF
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.service.lsa_service import LSAService


LOG = initialize_logger(__name__)


def start_lsa_run(hypothesis, model_connectivity, logger=None):

    if logger is None:
        logger = initialize_logger(__name__)

    logger.info("creating model configuration...")
    model_configuration_service = ModelConfigurationService(hypothesis.number_of_regions)
    model_configuration = model_configuration_service. \
        configure_model_from_hypothesis(hypothesis, model_connectivity)

    logger.info("running LSA...")
    lsa_service = LSAService(eigen_vectors_number_selection=EIGENVECTORS_NUMBER_SELECTION, eigen_vectors_number=None,
                             weighted_eigenvector_sum=WEIGHTED_EIGENVECTOR_SUM, normalize_propagation_strength=False)
    lsa_hypothesis = lsa_service.run_lsa(hypothesis, model_configuration)

    return model_configuration_service, model_configuration, lsa_service, lsa_hypothesis


def from_head_to_hypotheses(ep_name, data_mode=DATA_MODE, data_folder=os.path.join(DATA_CUSTOM, 'Head'),
                            plot_head=False, figure_dir=FOLDER_FIGURES, sensors_filename="SensorsInternal.h5",
                            logger=LOG):
    if data_mode is TVB:
        from tvb_epilepsy.tvb_api.readers_tvb import TVBReader as Reader
    else:
        from tvb_epilepsy.custom.readers_custom import CustomReader as Reader
    # -------------------------------Reading model_data-----------------------------------
    reader = Reader()
    logger.info("Reading from: " + data_folder)
    head = reader.read_head(data_folder, seeg_sensors_files=[(sensors_filename, "")])
    if plot_head:
        head.plot(figure_dir=figure_dir)
    # --------------------------Hypothesis definition-----------------------------------
    # # Manual definition of hypothesis...:
    # x0_indices = [20]
    # x0_values = [0.9]
    # e_indices = [70]
    # e_values = [0.9]
    # disease_values = x0_values + e_values
    # disease_indices = x0_indices + e_indices
    # ...or reading a custom file:
    # FOLDER_RES = os.path.join(data_folder, ep_name)
    disease_values = reader.read_epileptogenicity(data_folder, name=ep_name)
    disease_indices, = np.where(disease_values > np.min([X0_DEF, E_DEF]))
    disease_values = disease_values[disease_indices]
    inds = np.argsort(disease_values)
    disease_values = disease_values[inds]
    disease_indices = disease_indices[inds]
    x0_indices = [disease_indices[-1]]
    x0_values = [disease_values[-1]]
    e_indices = disease_indices[0:-1].tolist()
    e_values = disease_values[0:-1].tolist()
    disease_indices = list(disease_indices)
    n_x0 = len(x0_indices)
    n_e = len(e_indices)
    n_disease = len(disease_indices)
    all_regions_indices = np.array(range(head.number_of_regions))
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()
    n_healthy = len(healthy_indices)
    # This is an example of Excitability Hypothesis:
    hyp_x0 = DiseaseHypothesis(head.connectivity.number_of_regions,
                               excitability_hypothesis={tuple(disease_indices): disease_values},
                               epileptogenicity_hypothesis={}, connectivity_hypothesis={})
    # This is an example of Mixed Hypothesis:
    hyp_x0_E = DiseaseHypothesis(head.connectivity.number_of_regions,
                               excitability_hypothesis={tuple(x0_indices): x0_values},
                               epileptogenicity_hypothesis={tuple(e_indices): e_values}, connectivity_hypothesis={})
    hyp_E = DiseaseHypothesis(head.connectivity.number_of_regions,
                               excitability_hypothesis={},
                               epileptogenicity_hypothesis={tuple(disease_indices): disease_values}, connectivity_hypothesis={})
    hypos = (hyp_x0, hyp_E, hyp_x0_E)
    return head, hypos


def from_hypothesis_to_model_config_lsa(hyp, head, eigen_vectors_number=None, weighted_eigenvector_sum=True,
                                        plot_flag=True, save_flag=True, results_dir=FOLDER_RES,
                                        figure_dir=FOLDER_FIGURES, logger=LOG, **kwargs):
    logger.info("\n\nRunning hypothesis: " + hyp.name)
    # if save_flag:
    #     hyp.write_to_h5(results_dir, hyp.name + ".h5")
    logger.info("\n\nCreating model configuration...")
    model_configuration_service = ModelConfigurationService(hyp.number_of_regions, **kwargs)
    # if save_flag:
    #     model_configuration_service.write_to_h5(results_dir, hyp.name + "_model_config_service.h5")
    if hyp.type == "Epileptogenicity":
        model_configuration = model_configuration_service. \
            configure_model_from_E_hypothesis(hyp, head.connectivity.normalized_weights)
    else:
        model_configuration = model_configuration_service. \
            configure_model_from_hypothesis(hyp, head.connectivity.normalized_weights)
    if save_flag:
        model_configuration.write_to_h5(results_dir, hyp.name + "_ModelConfig.h5")
    # Plot nullclines and equilibria of model configuration
    if plot_flag:
        model_configuration_service.plot_nullclines_eq(model_configuration, head.connectivity.region_labels,
                                                   special_idx=hyp.get_regions_disease(), model="6d", zmode="lin",
                                                   figure_name=hyp.name + "_Nullclines and equilibria",
                                                   figure_dir=figure_dir)

    logger.info("\n\nRunning LSA...")
    lsa_service = LSAService(eigen_vectors_number=eigen_vectors_number,
                             weighted_eigenvector_sum=weighted_eigenvector_sum)
    lsa_hypothesis = lsa_service.run_lsa(hyp, model_configuration)
    if save_flag:
        lsa_hypothesis.write_to_h5(results_dir, lsa_hypothesis.name + "_LSA.h5")
        # lsa_service.write_to_h5(results_dir, lsa_hypothesis.name + "_LSAConfig.h5")
    if plot_flag:
        lsa_service.plot_lsa(lsa_hypothesis, model_configuration, head.connectivity.region_labels, None)
    return model_configuration, lsa_hypothesis, model_configuration_service, lsa_service