"""
Entry point for working with VEP
"""
import os
import numpy as np
from tvb_epilepsy.base.constants.config import Config, OutputConfig
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.io.tvb_data_reader import TVBReader
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.lsa_service import LSAService
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.top.scripts.pse_scripts import pse_from_lsa_hypothesis
from tvb_epilepsy.top.scripts.simulation_scripts import from_model_configuration_to_simulation


def main_cc_vep(config, head_folder, ep_name="clinical_hypothesis", x0_indices=[], pse_flag=False, sim_flag=True):
    if not (os.path.isdir(config.out.FOLDER_RES)):
        os.mkdir(config.out.FOLDER_RES)
    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)

    # -------------------------------Reading data-----------------------------------
    reader = TVBReader() if config.input.IS_TVB_MODE else H5Reader()
    writer = H5Writer()
    logger.info("Reading from: %s", head_folder)
    head = reader.read_head(head_folder)
    plotter = Plotter(config)
    plotter.plot_head(head)

    # --------------------------Hypothesis definition-----------------------------------
    hypo_builder = HypothesisBuilder().set_nr_of_regions(head.connectivity.number_of_regions)
    all_regions_indices = np.array(range(head.number_of_regions))

    # This is an example of Epileptogenicity Hypothesis:
    hyp_E = hypo_builder.build_hypothesis_from_file(ep_name, x0_indices)
    # This is an example of Excitability Hypothesis:
    hyp_x0 = hypo_builder.build_hypothesis_from_file(ep_name)

    disease_indices = hyp_E.e_indices + hyp_x0.x0_indices
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()

    if len(x0_indices) > 0:
        # This is an example of x0_values mixed Excitability and Epileptogenicity Hypothesis:
        disease_values = reader.read_epileptogenicity(head_folder, name=ep_name)
        disease_values = disease_values.tolist()
        x0_values = []
        for ix0 in x0_indices:
            ind = disease_indices.index(ix0)
            del disease_indices[ind]
            x0_values.append(disease_values.pop(ind))
        e_indices = disease_indices
        e_values = np.array(disease_values)
        x0_values = np.array(x0_values)
        hyp_x0_E = hypo_builder._build_mixed_hypothesis(e_values, e_indices, x0_values, x0_indices)
        hypotheses = (hyp_E, hyp_x0, hyp_x0_E)

    else:
        hypotheses = (hyp_E, hyp_x0,)

    # --------------------------Hypothesis and LSA-----------------------------------
    for hyp in hypotheses:
        logger.info("Running hypothesis: %s", hyp.name)
        logger.info("Creating model configuration...")
        builder = ModelConfigurationBuilder(hyp.number_of_regions)
        writer.write_model_configuration_builder(builder,
                                                 os.path.join(config.out.FOLDER_RES, "model_config_service.h5"))
        if hyp.type == "Epileptogenicity":
            model_configuration = builder.build_model_from_E_hypothesis(hyp, head.connectivity.normalized_weights)
        else:
            model_configuration = builder.build_model_from_hypothesis(hyp, head.connectivity.normalized_weights)
        writer.write_model_configuration(model_configuration,
                                         os.path.join(config.out.FOLDER_RES, "ModelConfiguration.h5"))
        # Plot nullclines and equilibria of model configuration
        plotter.plot_state_space(model_configuration, head.connectivity.region_labels,
                                 special_idx=disease_indices, model="2d", zmode="lin",
                                 figure_name=hyp.name + "_StateSpace")
        logger.info("Running LSA...")
        lsa_service = LSAService(eigen_vectors_number=None, weighted_eigenvector_sum=True)
        lsa_hypothesis = lsa_service.run_lsa(hyp, model_configuration)
        writer.write_hypothesis(lsa_hypothesis, os.path.join(config.out.FOLDER_RES, lsa_hypothesis.name + ".h5"))
        writer.write_lsa_service(lsa_service, os.path.join(config.out.FOLDER_RES, "lsa_config_service.h5"))
        plotter.plot_lsa(lsa_hypothesis, model_configuration, lsa_service.weighted_eigenvector_sum,
                         lsa_service.eigen_vectors_number, head.connectivity.region_labels, None)
        if pse_flag:
            n_samples = 100
            # --------------Parameter Search Exploration (PSE)-------------------------------
            logger.info("Running PSE LSA...")
            pse_results = pse_from_lsa_hypothesis(lsa_hypothesis,
                                                  head.connectivity.normalized_weights,
                                                  head.connectivity.region_labels,
                                                  n_samples, param_range=0.1,
                                                  global_coupling=[{"indices": all_regions_indices}],
                                                  healthy_regions_parameters=[{"name": "x0_values",
                                                                               "indices": healthy_indices}],
                                                  model_configuration_builder=builder,
                                                  lsa_service=lsa_service, save_flag=True,
                                                  folder_res=config.out.FOLDER_RES,
                                                  filename="PSE_LSA", logger=logger)[0]
            plotter.plot_lsa(lsa_hypothesis, model_configuration, lsa_service.weighted_eigenvector_sum,
                             lsa_service.eigen_vectors_number, head.connectivity.region_labels, pse_results,
                             title="Hypothesis PSE LSA Overview")
        if sim_flag:
            config.out.subfolder = "simulations"
            for folder in (config.out.FOLDER_RES, config.out.FOLDER_FIGURES):
                if not (os.path.isdir(folder)):
                    os.mkdir(folder)
            dynamical_models = ["EpileptorDP2D", "EpileptorDPrealistic"]

            for dynamical_model, sim_type in zip(dynamical_models, ["fitting", "realistic"]):
                ts_file = None  # os.path.join(sim_folder_res, dynamical_model + "_ts.h5")
                vois_ts_dict = \
                    from_model_configuration_to_simulation(model_configuration, head, lsa_hypothesis,
                                                           sim_type=sim_type, dynamical_model=dynamical_model,
                                                           ts_file=ts_file, plot_flag=True, config=config)


if __name__ == "__main__":

    subjects_top_folder = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', 'results', "CC")
    subject_base_name = "TVB"
    subj_ids = [1, 2, 3, 4, 4]

    x0_indices = ([40, 42], [], [1, 26], [], [])
    ep_names = (3 * ["clinical_hypothesis_preseeg"] + ["clinical_hypothesis_preseeg_right"]
                + ["clinical_hypothesis_preseeg_bilateral"])

    for subj_id in range(4, len(subj_ids)):
        subject_name = subject_base_name + str(subj_ids[subj_id])
        head_path = os.path.join(subjects_top_folder, subject_name, "Head")
        x0_inds = x0_indices[subj_id]
        folder_results = os.path.join(head_path, ep_names[subj_id], "res")

        config = Config(head_path, Config.generic.MODE_JAVA, folder_results, True)

        main_cc_vep(config, head_folder=head_path, ep_name=ep_names[subj_id], x0_indices=x0_inds)
