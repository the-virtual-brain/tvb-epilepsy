"""
Entry point for working with VEP
"""
import os

import numpy as np

from tvb_epilepsy.base.constants.module_constants import TVB, DATA_MODE
from tvb_epilepsy.base.constants.configurations import FOLDER_VEP, FOLDER_RES
from tvb_epilepsy.base.constants.model_constants import X0_DEF, E_DEF
from tvb_epilepsy.base.h5_model import convert_to_h5_model, read_h5_model
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.service.lsa_service import LSAService
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.scripts.pse_scripts import pse_from_lsa_hypothesis
from tvb_epilepsy.scripts.simulation_scripts import from_model_configuration_to_simulation

if DATA_MODE is TVB:
    from tvb_epilepsy.tvb_api.readers_tvb import TVBReader as Reader
else:
    from tvb_epilepsy.custom.readers_custom import CustomReader as Reader


PSE_FLAG = False
SIM_FLAG = True


HEAD="Head"


def main_vep(subject="TVB3", ep_name="clinical_hypothesis", x0_indices=[], folder_res=FOLDER_RES,
             pse_flag=PSE_FLAG, sim_flag=SIM_FLAG):

    subject_folder = os.path.join(FOLDER_VEP, subject)
    folder_res = os.path.basename(folder_res)
    FOLDER_RES = os.path.join(subject_folder, HEAD, ep_name)
    if not (os.path.isdir(FOLDER_RES)):
        os.mkdir(FOLDER_RES)
    FOLDER_RES = os.path.join(FOLDER_RES, folder_res)
    if not (os.path.isdir(FOLDER_RES)):
        os.mkdir(FOLDER_RES)
    FOLDER_FIGS = os.path.join(FOLDER_RES, "figs")
    FOLDER_LOGS = os.path.join(FOLDER_RES, "logs")
    for FOLDER in (FOLDER_RES, FOLDER_FIGS, FOLDER_LOGS):
        if not (os.path.isdir(FOLDER)):
            os.mkdir(FOLDER)

    logger = initialize_logger(__name__, FOLDER_LOGS)

    # -------------------------------Reading data-----------------------------------
    data_folder = os.path.join(subject_folder, HEAD)
    reader = Reader()
    logger.info("Reading from: " + data_folder)
    head = reader.read_head(data_folder)
    # head.plot(figure_dir=FOLDER_FIGS)
    # --------------------------Hypothesis definition-----------------------------------
    # # Manual definition of hypothesis...:
    # x0_indices = [20]
    # x0_values = [0.9]
    # e_indices = [70]
    # e_values = [0.9]
    # disease_values = x0_values + e_values
    # disease_indices = x0_indices + e_indices
    # ...or reading a custom file:
    from tvb_epilepsy.custom.readers_custom import CustomReader
    if not isinstance(reader, CustomReader):
        reader = CustomReader()
    disease_values = reader.read_epileptogenicity(data_folder, name=ep_name)
    disease_indices, = np.where(disease_values > np.min([X0_DEF, E_DEF]))
    disease_values = disease_values[disease_indices]
    # TODO: something smarter to normalize better disease values
    disease_values += (0.95 - np.max(disease_values))
    disease_indices = list(disease_indices)
    n_x0 = len(x0_indices)
    # n_e = len(e_indices)
    n_disease = len(disease_indices)
    all_regions_indices = np.array(range(head.number_of_regions))
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()
    n_healthy = len(healthy_indices)
    # This is an example of Epileptogenicity Hypothesis:
    hyp_E = DiseaseHypothesis(head.connectivity.number_of_regions,
                              excitability_hypothesis={},
                              epileptogenicity_hypothesis={tuple(disease_indices): disease_values},
                              connectivity_hypothesis={})
    hypotheses = (hyp_E,)
    # # This is an example of Excitability Hypothesis:
    hyp_x0 = DiseaseHypothesis(head.connectivity.number_of_regions,
                               excitability_hypothesis={tuple(disease_indices): disease_values},
                               epileptogenicity_hypothesis={}, connectivity_hypothesis={})

    if n_x0 > 0:
        # This is an example of x0_values mixed Excitability and Epileptogenicity Hypothesis:
        disease_values = disease_values.tolist()
        x0_values = []
        for ix0 in x0_indices:
            ind = disease_indices.index(ix0)
            del disease_indices[ind]
            x0_values.append(disease_values.pop(ind))
        e_indices = disease_indices
        e_values = np.array(disease_values)
        x0_values = np.array(x0_values)
        hyp_x0_E = DiseaseHypothesis(head.connectivity.number_of_regions,
                                     excitability_hypothesis={tuple(x0_indices): x0_values},
                                     epileptogenicity_hypothesis={tuple(e_indices): e_values},
                                     connectivity_hypothesis={})
        hypotheses = (hyp_E, hyp_x0, hyp_x0_E)

    else:
        hypotheses = (hyp_E, hyp_x0,)

    # --------------------------Hypothesis and LSA-----------------------------------
    for hyp in hypotheses:
        folder_res = os.path.join(FOLDER_RES, hyp.name)
        folder_figs = os.path.join(FOLDER_FIGS, hyp.name)
        for folder in (folder_res, folder_figs):
            if not (os.path.isdir(folder)):
                os.mkdir(folder)
        logger.info("\n\nRunning hypothesis: " + hyp.name)
        # hyp.write_to_h5(FOLDER_RES, hyp.name + ".h5")
        logger.info("\n\nCreating model configuration...")
        model_configuration_service = ModelConfigurationService(hyp.number_of_regions)
        model_configuration_service.write_to_h5(folder_res, "model_config_service.h5")
        if hyp.type == "Epileptogenicity":
            model_configuration = model_configuration_service.\
                                            configure_model_from_E_hypothesis(hyp, head.connectivity.normalized_weights)
        else:
            model_configuration = model_configuration_service.\
                                              configure_model_from_hypothesis(hyp, head.connectivity.normalized_weights)
        model_configuration.write_to_h5(folder_res, "ModelConfiguration.h5")
        # Plot nullclines and equilibria of model configuration
        model_configuration_service.plot_state_space(model_configuration, head.connectivity.region_labels,
                                                     special_idx=disease_indices, model="2d", zmode="lin",
                                                     figure_name=hyp.name + "_StateSpace", figure_dir=folder_figs)
        logger.info("\n\nRunning LSA...")
        lsa_service = LSAService(eigen_vectors_number=None, weighted_eigenvector_sum=True)
        lsa_hypothesis = lsa_service.run_lsa(hyp, model_configuration)
        lsa_hypothesis.write_to_h5(folder_res, lsa_hypothesis.name + ".h5")
        lsa_service.write_to_h5(folder_res, "lsa_config_service.h5")
        lsa_service.plot_lsa(lsa_hypothesis, model_configuration, head.connectivity.region_labels,  None,
                             figure_dir=folder_figs)
        if pse_flag:
            n_samples = 100
            #--------------Parameter Search Exploration (PSE)-------------------------------
            logger.info("\n\nRunning PSE LSA...")
            pse_results = pse_from_lsa_hypothesis(lsa_hypothesis,
                                                  head.connectivity.normalized_weights,
                                                  head.connectivity.region_labels,
                                                  n_samples, param_range=0.1,
                                                  global_coupling=[{"indices": all_regions_indices}],
                                                  healthy_regions_parameters=[{"name": "x0_values",
                                                                               "indices": healthy_indices}],
                                                  model_configuration_service=model_configuration_service,
                                                  lsa_service=lsa_service, save_flag=True, folder_res=folder_res,
                                                  filename="PSE_LSA", logger=logger)[0]
            lsa_service.plot_lsa(lsa_hypothesis, model_configuration, head.connectivity.region_labels, pse_results,
                                 title="Hypothesis PSE LSA Overview", figure_dir=folder_figs)
        if sim_flag:
            sim_folder_res = os.path.join(folder_res, "simulations")
            sim_folder_figs = os.path.join(folder_figs, "simulations")
            for folder in (sim_folder_res, sim_folder_figs):
                if not (os.path.isdir(folder)):
                    os.mkdir(folder)
            dynamical_models = ["EpileptorDP2D", "EpileptorDPrealistic"]

            for dynamical_model, sim_type in zip(dynamical_models, ["fitting", "realistic"]):
                ts_file = None  # os.path.join(sim_folder_res, dynamical_model + "_ts.h5")
                vois_ts_dict = \
                    from_model_configuration_to_simulation(model_configuration, head, lsa_hypothesis,
                                                           sim_type=sim_type, dynamical_model=dynamical_model,
                                                           simulation_mode=TVB, ts_file=ts_file, plot_flag=True,
                                                           results_dir=sim_folder_res, figure_dir=sim_folder_figs,
                                                           logger=logger)


if __name__ == "__main__":

    x0_indices = ([40, 42], [], [1, 26], [], [])

    SUBJECT = "TVB"

    subj_ids = [1, 2, 3, 4, 4]
    ep_names = 3 * ["clinical_hypothesis_preseeg"]\
               + ["clinical_hypothesis_preseeg_right"] + ["clinical_hypothesis_preseeg_bilateral"]
    for subj_id in range(4, len(subj_ids)):
        subject = SUBJECT + str(subj_ids[subj_id])
        ep_name = ep_names[subj_id]
        x0_inds = x0_indices[subj_id]
        main_vep(subject=subject, ep_name=ep_name, x0_indices=x0_inds, folder_res=FOLDER_RES,
                 pse_flag=PSE_FLAG, sim_flag=SIM_FLAG)
