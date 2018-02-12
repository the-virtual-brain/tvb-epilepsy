import os
import numpy as np
from tvb_epilepsy.base.constants.configurations import HEAD_FOLDER, FOLDER_RES, FOLDER_FIGURES
from tvb_epilepsy.base.constants.module_constants import TVB, DATA_MODE, EIGENVECTORS_NUMBER_SELECTION, \
    WEIGHTED_EIGENVECTOR_SUM
from tvb_epilepsy.base.constants.model_constants import X0_DEF, E_DEF
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationBuilder
from tvb_epilepsy.service.lsa_service import LSAService

logger = initialize_logger(__name__)


def start_lsa_run(hypothesis, model_connectivity):
    logger.info("creating model configuration...")
    model_configuration_service = ModelConfigurationBuilder(hypothesis.number_of_regions)
    model_configuration = model_configuration_service. \
        build_model_from_hypothesis(hypothesis, model_connectivity)

    logger.info("running LSA...")
    lsa_service = LSAService(eigen_vectors_number_selection=EIGENVECTORS_NUMBER_SELECTION, eigen_vectors_number=None,
                             weighted_eigenvector_sum=WEIGHTED_EIGENVECTOR_SUM, normalize_propagation_strength=False)
    lsa_hypothesis = lsa_service.run_lsa(hypothesis, model_configuration)

    return model_configuration_service, model_configuration, lsa_service, lsa_hypothesis


def from_head_to_hypotheses(ep_name, data_mode=DATA_MODE, data_folder=HEAD_FOLDER,
                            plot_head=False, figure_dir=FOLDER_FIGURES, sensors_filename="SensorsInternal.h5"):
    if data_mode is TVB:
        from tvb_epilepsy.io.tvb_data_reader import TVBReader as Reader
    else:
        from tvb_epilepsy.io.h5_reader import H5Reader as Reader
    # -------------------------------Reading model_data-----------------------------------
    reader = Reader()
    logger.info("Reading from: " + data_folder)
    head = reader.read_head(data_folder)
    if plot_head:
        plotter = Plotter()
        plotter.plot_head(head, figure_dir=figure_dir)
        # head.plot(figure_dir=figure_dir)
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

    hypo_builder = HypothesisBuilder().set_nr_of_regions(head.connectivity.number_of_regions).set_sort_disease_values(
        True)
    threshold = np.min([X0_DEF, E_DEF])

    # This is an example of Excitability Hypothesis:
    hyp_x0 = hypo_builder.build_excitability_hypothesis_based_on_threshold(disease_values, threshold)

    # This is an example of Mixed Hypothesis:
    hyp_x0_E = hypo_builder.build_mixed_hypothesis_with_x0_having_max_values(disease_values, threshold)

    # This is an example of Epileptogenicity Hypothesis:
    hyp_E = hypo_builder.build_epileptogenicity_hypothesis_based_on_threshold(disease_values, threshold)

    hypos = (hyp_x0, hyp_E, hyp_x0_E)
    return head, hypos


def from_hypothesis_to_model_config_lsa(hyp, head, eigen_vectors_number=None, weighted_eigenvector_sum=True,
                                        plot_flag=True, save_flag=True, results_dir=FOLDER_RES,
                                        figure_dir=FOLDER_FIGURES, **kwargs):
    logger.info("\n\nRunning hypothesis: " + hyp.name)
    logger.info("\n\nCreating model configuration...")
    model_configuration_service = ModelConfigurationBuilder(hyp.number_of_regions, **kwargs)
    if hyp.type == "Epileptogenicity":
        model_configuration = model_configuration_service. \
            build_model_from_E_hypothesis(hyp, head.connectivity.normalized_weights)
    else:
        model_configuration = model_configuration_service. \
            build_model_from_hypothesis(hyp, head.connectivity.normalized_weights)
    writer = H5Writer()
    if save_flag:
        writer.write_model_configuration(model_configuration, os.path.join(results_dir, hyp.name + "_ModelConfig.h5"))
    # Plot nullclines and equilibria of model configuration
    plotter = Plotter()
    if plot_flag:
        plotter.plot_state_space(model_configuration, head.connectivity.region_labels,
                                 special_idx=hyp.get_regions_disease(), model="6d", zmode="lin",
                                 figure_name=hyp.name + "_StateSpace", figure_dir=figure_dir)

    logger.info("\n\nRunning LSA...")
    lsa_service = LSAService(eigen_vectors_number=eigen_vectors_number,
                             weighted_eigenvector_sum=weighted_eigenvector_sum)
    lsa_hypothesis = lsa_service.run_lsa(hyp, model_configuration)
    if save_flag:
        writer.write_hypothesis(lsa_hypothesis, os.path.join(results_dir, lsa_hypothesis.name + "_LSA.h5"))
    if plot_flag:
        plotter.plot_lsa(lsa_hypothesis, model_configuration, lsa_service.weighted_eigenvector_sum,
                         lsa_service.eigen_vectors_number, head.connectivity.region_labels, None)
    return model_configuration, lsa_hypothesis, model_configuration_service, lsa_service
