"""
Entry point for working with VEP
"""
import os
import warnings
from copy import deepcopy

import numpy as np

from tvb_epilepsy.base.constants import FOLDER_RES, SIMULATION_MODE, TVB, DATA_MODE, VOIS, DATA_CUSTOM, X0_DEF, E_DEF
from tvb_epilepsy.base.h5_model import convert_to_h5_model, read_h5_model
from tvb_epilepsy.base.helper_functions import pse_from_lsa_hypothesis, sensitivity_analysis_pse_from_lsa_hypothesis, \
    set_time_scales
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.plot_utils import plot_sim_results
from tvb_epilepsy.base.service.lsa_service import LSAService
from tvb_epilepsy.base.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.base.utils import assert_equal_objects
from tvb_epilepsy.base.utils import initialize_logger, calculate_projection
from tvb_epilepsy.base.computations.analyzers_utils import filter_data
from tvb_epilepsy.custom.read_write import write_ts_epi, write_ts_seeg_epi
from tvb_epilepsy.custom.simulator_custom import EpileptorModel
from tvb_epilepsy.tvb_api.epileptor_models import EpileptorDP2D

if DATA_MODE is TVB:
    from tvb_epilepsy.tvb_api.readers_tvb import TVBReader as Reader
else:
    from tvb_epilepsy.custom.readers_custom import CustomReader as Reader

if SIMULATION_MODE is TVB:
    from tvb_epilepsy.base.helper_functions import setup_TVB_simulation_from_model_configuration \
        as setup_simulation_from_model_configuration
else:
    from tvb_epilepsy.custom.simulator_custom import setup_custpm_simulation_from_model_configuration \
        as setup_simulation_from_model_configuration


def prepare_vois_ts_dict(vois, data):
    # Pack results into a dictionary:
    vois_ts_dict = dict()
    for idx_voi, voi in enumerate(vois):
        vois_ts_dict[voi] = data[:, idx_voi, :].astype('f')

    return vois_ts_dict


def prepare_ts_and_seeg_h5_file(folder, filename, model, projections, vois_ts_dict, hpf_flag, hpf_low, hpf_high, fsAVG,
                                dt):
    # High pass filter, and compute SEEG:
    if isinstance(model, EpileptorDP2D):
        raw_data = np.dstack(
            [vois_ts_dict["x1"], vois_ts_dict["z"], vois_ts_dict["x1"]])
        lfp_data = vois_ts_dict["x1"]

        for idx_proj, proj in enumerate(projections):
            vois_ts_dict['seeg%d' % idx_proj] = vois_ts_dict['z'].dot(proj.T)

    else:
        if isinstance(model, EpileptorModel):
            lfp_data = vois_ts_dict["x2"] - vois_ts_dict["x1"]

        else:
            lfp_data = vois_ts_dict["lfp"]

        raw_data = np.dstack(
            [vois_ts_dict["x1"], vois_ts_dict["z"], vois_ts_dict["x2"]])

        for idx_proj, proj in enumerate(projections):
            vois_ts_dict['seeg%d' % idx_proj] = vois_ts_dict['lfp'].dot(proj.T)
            if hpf_flag:
                for i in range(vois_ts_dict['seeg'].shape[0]):
                    vois_ts_dict['seeg_hpf%d' % i][:, i] = filter_data(
                        vois_ts_dict['seeg%d' % i][:, i], hpf_low, hpf_high,
                        fsAVG)
    # Write files:
    write_ts_epi(raw_data, dt, lfp_data, folder, filename)

    for i in range(len(projections)):
        write_ts_seeg_epi(vois_ts_dict['seeg%d' % i], dt, folder, filename)


def main_vep(test_write_read=False):

    logger = initialize_logger(__name__)

    # -------------------------------Reading data-----------------------------------

    data_folder = os.path.join(DATA_CUSTOM, 'Head')

    reader = Reader()

    logger.info("Reading from: " + data_folder)
    head = reader.read_head(data_folder)

    # --------------------------Hypothesis definition-----------------------------------

    n_samples = 100

    # # Manual definition of hypothesis...:
    # x0_indices = [20]
    # x0_values = [0.9]
    # e_indices = [70]
    # e_values = [0.9]
    # disease_values = x0_values + e_values
    # disease_indices = x0_indices + e_indices

    # ...or reading a custom file:
    ep_name = "ep_test1"
    #FOLDER_RES = os.path.join(data_folder, ep_name)
    from tvb_epilepsy.custom.readers_custom import CustomReader

    if not isinstance(reader, CustomReader):
        reader = CustomReader()
    disease_values = reader.read_epileptogenicity(data_folder, name=ep_name)
    disease_indices, = np.where(disease_values > np.min([X0_DEF, E_DEF]))
    disease_values = disease_values[disease_indices]
    if disease_values.size > 1:
        inds_split = np.ceil(disease_values.size * 1.0 / 2).astype("int")
        x0_indices = disease_indices[:inds_split].tolist()
        e_indices = disease_indices[inds_split:].tolist()
        x0_values = disease_values[:inds_split].tolist()
        e_values = disease_values[inds_split:].tolist()
    else:
        x0_indices = disease_indices.tolist()
        x0_values = disease_values.tolist()
        e_indices = []
        e_values = []
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

    # This is an example of Epileptogenicity Hypothesis:
    hyp_E = DiseaseHypothesis(head.connectivity.number_of_regions,
                              excitability_hypothesis={},
                              epileptogenicity_hypothesis={tuple(disease_indices): disease_values},
                              connectivity_hypothesis={})

    if len(e_indices) > 0:
        # This is an example of x0 mixed Excitability and Epileptogenicity Hypothesis:
        hyp_x0_E = DiseaseHypothesis(head.connectivity.number_of_regions,
                                     excitability_hypothesis={tuple(x0_indices): x0_values},
                                     epileptogenicity_hypothesis={tuple(e_indices): e_values},
                                     connectivity_hypothesis={})
        hypotheses = (hyp_x0, hyp_E, hyp_x0_E)
    else:
        hypotheses = (hyp_x0, hyp_E)

    # --------------------------Projections computations-----------------------------------

    sensorsSEEG = []
    projections = []
    for sensors, projection in head.sensorsSEEG.iteritems():
        if projection is None:
            continue
        else:
            projection = calculate_projection(sensors, head.connectivity)
            head.sensorsSEEG[sensors] = projection
            sensorsSEEG.append(sensors)
            projections.append(projection)

    # --------------------------Simulation preparations-----------------------------------

    # TODO: maybe use a custom Monitor class
    fs = 2 * 4096.0
    scale_time = 2.0
    time_length = 10000.0
    scale_fsavg = 2.0
    report_every_n_monitor_steps = 10.0
    (dt, fsAVG, sim_length, monitor_period, n_report_blocks) = \
        set_time_scales(fs=fs, dt=None, time_length=time_length, scale_time=scale_time, scale_fsavg=scale_fsavg,
                        report_every_n_monitor_steps=report_every_n_monitor_steps)

    model_name = "EpileptorDP"

    # We don't want any time delays for the moment
    head.connectivity.tract_lengths *= 0.0

    hpf_flag = False
    hpf_low = max(16.0, 1000.0 / time_length)  # msec
    hpf_high = min(250.0, fsAVG)

    # --------------------------Hypothesis and LSA-----------------------------------

    for hyp in hypotheses:

        logger.info("\n\nRunning hypothesis: " + hyp.name)

        # hyp.write_to_h5(FOLDER_RES, hyp.name + ".h5")

        logger.info("\n\nCreating model configuration...")
        model_configuration_service = ModelConfigurationService(hyp.number_of_regions)
        model_configuration_service.write_to_h5(FOLDER_RES, hyp.name + "_model_config_service.h5")
        if test_write_read:
            logger.info("Written and read model configuration services are identical?: "+
                        str(assert_equal_objects(model_configuration_service,
                                 read_h5_model(os.path.join(FOLDER_RES, hyp.name + "_model_config_service.h5")).
                                    convert_from_h5_model(obj=deepcopy(model_configuration_service)))))

        if hyp.type == "Epileptogenicity":
            model_configuration = model_configuration_service.\
                                            configure_model_from_E_hypothesis(hyp, head.connectivity.normalized_weights)
        else:
            model_configuration = model_configuration_service.\
                                              configure_model_from_hypothesis(hyp, head.connectivity.normalized_weights)
        model_configuration.write_to_h5(FOLDER_RES, hyp.name + "_ModelConfig.h5")
        if test_write_read:
            logger.info("Written and read model configuration are identical?: " +
                        str(assert_equal_objects(model_configuration,
                                                 read_h5_model(os.path.join(FOLDER_RES, hyp.name + "_ModelConfig.h5")).
                                                 convert_from_h5_model(obj=deepcopy(model_configuration)))))

        # # Plot nullclines and equilibria of model configuration
        # model_configuration_service.plot_nullclines_eq(model_configuration, head.connectivity.region_labels,
        #                                        special_idx=lsa_hypothesis.propagation_indices,
        #                                        model=str(model.nvar) + "d", zmode=model.zmode,
        #                                        figure_name=lsa_hypothesis.name + "_Nullclines and equilibria")

        logger.info("\n\nRunning LSA...")
        lsa_service = LSAService(eigen_vectors_number=None, weighted_eigenvector_sum=True)
        lsa_hypothesis = lsa_service.run_lsa(hyp, model_configuration)

        lsa_hypothesis.write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_LSA.h5")
        lsa_service.write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_LSAConfig.h5")
        if test_write_read:

            hypothesis_template = DiseaseHypothesis(hyp.number_of_regions)

            logger.info("Written and read LSA services are identical?: " +
                        str(assert_equal_objects(lsa_service,
                                 read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_LSAConfig.h5")).
                                    convert_from_h5_model(obj=deepcopy(lsa_service)))))
            logger.info("Written and read LSA hypotheses are identical (input object check)?: " +
                        str(assert_equal_objects(lsa_hypothesis,
                                 read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_LSA.h5")).
                                    convert_from_h5_model(obj=deepcopy(lsa_hypothesis)))))
            logger.info("Written and read LSA hypotheses are identical (input template check)?: " +
                        str(assert_equal_objects(lsa_hypothesis,
                                read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_LSA.h5")).
                                    convert_from_h5_model(obj=hypothesis_template))))
            logger.info("Written and read LSA hypotheses are identical (no input object check)?: " +
                        str(assert_equal_objects(pse_results,
                                read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_PSE_LSA_results.h5")).
                                    convert_from_h5_model())))
            logger.info("Written and read LSA services are identical?: " +
                        str(assert_equal_objects(lsa_service,
                                 read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_LSAConfig.h5")).
                                    convert_from_h5_model(obj=deepcopy(lsa_service)))))

        lsa_service.plot_lsa(lsa_hypothesis, model_configuration, head.connectivity.region_labels,  None,
                             title="LSA overview " + lsa_hypothesis.name)

        #--------------Parameter Search Exploration (PSE)-------------------------------

        logger.info("\n\nRunning PSE LSA...")
        pse_results = pse_from_lsa_hypothesis(lsa_hypothesis,
                                          head.connectivity.normalized_weights,
                                          head.connectivity.region_labels,
                                          n_samples, half_range=0.1,
                                          global_coupling=[{"indices": all_regions_indices}],
                                          healthy_regions_parameters=[{"name": "x0", "indices": healthy_indices}],
                                          model_configuration_service=model_configuration_service,
                                          lsa_service=lsa_service, logger=logger)[0]

        lsa_service.plot_lsa(lsa_hypothesis, model_configuration, head.connectivity.region_labels, pse_results,
                             title="PSE LSA overview " + lsa_hypothesis.name)
        # , show_flag=True, save_flag=False

        convert_to_h5_model(pse_results).write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_PSE_LSA_results.h5")
        if test_write_read:
            logger.info("Written and read sensitivity analysis parameter search results are identical?: " +
                        str(assert_equal_objects(pse_results,
                                  read_h5_model(os.path.join(FOLDER_RES,
                                           lsa_hypothesis.name + "_PSE_LSA_results.h5")).convert_from_h5_model())))


        # --------------Sensitivity Analysis Parameter Search Exploration (PSE)-------------------------------

        logger.info("\n\nrunning sensitivity analysis PSE LSA...")
        sa_results, pse_sa_results = \
            sensitivity_analysis_pse_from_lsa_hypothesis(lsa_hypothesis,
                                                     head.connectivity.normalized_weights,
                                                     head.connectivity.region_labels,
                                                     n_samples, method="sobol", half_range=0.1,
                                     global_coupling=[{"indices": all_regions_indices,
                                                       "bounds":[0.0, 2 * model_configuration_service.K_unscaled[ 0]]}],
                                     healthy_regions_parameters=[{"name": "x0", "indices": healthy_indices}],
                                     model_configuration_service=model_configuration_service, lsa_service=lsa_service,
                                    logger=logger)


        lsa_service.plot_lsa(lsa_hypothesis, model_configuration, head.connectivity.region_labels, pse_sa_results,
                                    title="SA PSE LSA overview " + lsa_hypothesis.name)
        # , show_flag=True, save_flag=False

        convert_to_h5_model(pse_sa_results).write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_SA_PSE_LSA_results.h5")
        convert_to_h5_model(sa_results).write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_SA_LSA_results.h5")
        if test_write_read:
            logger.info("Written and read sensitivity analysis results are identical?: " +
                        str(assert_equal_objects(sa_results,
                                  read_h5_model(os.path.join(FOLDER_RES,
                                              lsa_hypothesis.name + "_SA_LSA_results.h5")).convert_from_h5_model())))
            logger.info("Written and read sensitivity analysis parameter search results are identical?: " +
                        str(assert_equal_objects(pse_results,
                                  read_h5_model(os.path.join(FOLDER_RES,
                                            lsa_hypothesis.name + "_SA_PSE_LSA_results.h5")).convert_from_h5_model())))
            logger.info("Written and read simulation settings are identical?: " +
                        str(assert_equal_objects(sim.simulation_settings,
                                                 read_h5_model(os.path.join(FOLDER_RES,
                                                                            lsa_hypothesis.name + "_sim_settings.h5")).
                                                 convert_from_h5_model(obj=deepcopy(sim.simulation_settings)))))

        # ------------------------------Simulation--------------------------------------
        logger.info("\n\nSimulating...")
        sim = setup_simulation_from_model_configuration(model_configuration, head.connectivity, dt,
                                                                       sim_length, monitor_period, model_name,
                                                                       scale_time=scale_time, noise_intensity=10 ** -8)

        sim.config_simulation()
        ttavg, tavg_data, status = sim.launch_simulation(n_report_blocks)

        convert_to_h5_model(sim.simulation_settings).write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_sim_settings.h5")
        if test_write_read:
            logger.info("Written and read sensitivity analysis results are identical?: " +
                        str(assert_equal_objects(sa_results,
                                  read_h5_model(os.path.join(FOLDER_RES,
                                              lsa_hypothesis.name + "_SA_LSA_results.h5")).convert_from_h5_model())))
            logger.info("Written and read sensitivity analysis parameter search results are identical?: " +
                        str(assert_equal_objects(pse_results,
                                  read_h5_model(os.path.join(FOLDER_RES,
                                            lsa_hypothesis.name + "_SA_PSE_LSA_results.h5")).convert_from_h5_model())))
            logger.info("Written and read simulation settings are identical?: " +
                        str(assert_equal_objects(sim.simulation_settings,
                                                 read_h5_model(os.path.join(FOLDER_RES,
                                                                            lsa_hypothesis.name + "_sim_settings.h5")).
                                                 convert_from_h5_model(obj=deepcopy(sim.simulation_settings)))))

        if not status:
            warnings.warn("\nSimulation failed!")

        else:

            tavg_data = tavg_data[:, :, :, 0]

            vois = VOIS[model_name]

            model = sim.model

            logger.info("\n\nSimulated signal return shape: %s", tavg_data.shape)
            logger.info("Time: %s - %s", scale_time * ttavg[0], scale_time * ttavg[-1])
            logger.info("Values: %s - %s", tavg_data.min(), tavg_data.max())

            time = scale_time * np.array(ttavg, dtype='float32')
            sampling_time = np.min(np.diff(time))

            vois_ts_dict = prepare_vois_ts_dict(vois, tavg_data)

            prepare_ts_and_seeg_h5_file(FOLDER_RES, lsa_hypothesis.name + "_ts.h5", model, projections, vois_ts_dict,
                                        hpf_flag, hpf_low, hpf_high, fsAVG, sampling_time)

            vois_ts_dict['time'] = time

            # Plot results
            plot_sim_results(model, lsa_hypothesis.propagation_indices, lsa_hypothesis.name, head, vois_ts_dict,
                             sensorsSEEG, hpf_flag)

            # Save results
            vois_ts_dict['time_units'] = 'msec'
            # savemat(os.path.join(FOLDER_RES, hypothesis.name + "_ts.mat"), vois_ts_dict)

        # if test_write_read:
        #
        #     hypothesis_template = DiseaseHypothesis(hyp.number_of_regions)
        #
        #     logger.info("Written and read model configuration services are identical?: "+
        #                 str(assert_equal_objects(model_configuration_service,
        #                          read_h5_model(os.path.join(FOLDER_RES, hyp.name + "_model_config_service.h5")).
        #                             convert_from_h5_model(obj=deepcopy(model_configuration_service)))))
        #     logger.info("Written and read model configuration are identical?: " +
        #                 str(assert_equal_objects(model_configuration,
        #                          read_h5_model(os.path.join(FOLDER_RES, hyp.name + "_ModelConfig.h5")).
        #                             convert_from_h5_model(obj=deepcopy(model_configuration)))))
        #     logger.info("Written and read LSA services are identical?: " +
        #                 str(assert_equal_objects(lsa_service,
        #                          read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_LSAConfig.h5")).
        #                             convert_from_h5_model(obj=deepcopy(lsa_service)))))
        #     logger.info("Written and read LSA hypotheses are identical (input object check)?: " +
        #                 str(assert_equal_objects(lsa_hypothesis,
        #                          read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_LSA.h5")).
        #                             convert_from_h5_model(obj=deepcopy(lsa_hypothesis)))))
        #     logger.info("Written and read LSA hypotheses are identical (input template check)?: " +
        #                 str(assert_equal_objects(lsa_hypothesis,
        #                         read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_LSA.h5")).
        #                             convert_from_h5_model(obj=hypothesis_template))))
        #     logger.info("Written and read LSA hypotheses are identical (no input object check)?: " +
        #                 str(assert_equal_objects(pse_results,
        #                         read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_PSE_LSA_results.h5")).
        #                             convert_from_h5_model())))
        #     logger.info("Written and read sensitivity analysis results are identical?: " +
        #                 str(assert_equal_objects(sa_results,
        #                         read_h5_model(os.path.join(FOLDER_RES,
        #                                     lsa_hypothesis.name + "_SA_LSA_results.h5")).convert_from_h5_model())))
        #     logger.info("Written and read sensitivity analysis parameter search results are identical?: " +
        #                 str(assert_equal_objects(pse_sa_results,
        #                         read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_SA_PSE_LSA_results.h5")).
        #                             convert_from_h5_model())))
        #     logger.info("Written and read simulation settings are identical?: " +
        #                 str(assert_equal_objects(sim.simulation_settings,
        #                          read_h5_model(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_sim_settings.h5")).
        #                          convert_from_h5_model(obj=deepcopy(sim.simulation_settings)))))



if __name__ == "__main__":
    main_vep()