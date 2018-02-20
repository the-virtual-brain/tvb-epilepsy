"""
Entry point for working with VEP
"""
import os
import numpy as np
from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.constants.model_constants import COLORED_NOISE
from tvb_epilepsy.base.utils.data_structures_utils import assert_equal_objects, isequal_string
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.simulator_builder import SimulatorBuilder
from tvb_epilepsy.top.scripts.pse_scripts import pse_from_lsa_hypothesis
from tvb_epilepsy.top.scripts.sensitivity_analysis_sripts import sensitivity_analysis_pse_from_lsa_hypothesis
from tvb_epilepsy.top.scripts.simulation_scripts import prepare_vois_ts_dict
from tvb_epilepsy.top.scripts.simulation_scripts import compute_seeg_and_write_ts_h5_file
from tvb_epilepsy.service.lsa_service import LSAService
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.io.tvb_data_reader import TVBReader

PSE_FLAG = False
SA_PSE_FLAG = False
SIM_FLAG = True
EP_NAME = "ep_l_frontal_complex"


def main_vep(config=Config(), sim_type="default", test_write_read=False,
             pse_flag=PSE_FLAG, sa_pse_flag=SA_PSE_FLAG, sim_flag=SIM_FLAG):
    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
    # -------------------------------Reading data-----------------------------------
    reader = TVBReader() if config.input.IS_TVB_MODE else H5Reader()
    writer = H5Writer()
    logger.info("Reading from: " + config.input.HEAD)
    head = reader.read_head(config.input.HEAD)
    plotter = Plotter(config)
    plotter.plot_head(head)
    if test_write_read:
        writer.write_head(head, os.path.join(config.out.FOLDER_RES, "Head"))
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

    hypo_builder = HypothesisBuilder(config=config). \
        set_nr_of_regions(head.connectivity.number_of_regions). \
        set_sort_disease_values(True)

    # This is an example of Epileptogenicity Hypothesis: you give as ep all indices for values > 0
    hyp_E = hypo_builder.build_hypothesis_from_file(EP_NAME, [1, 3, 16, 25])

    # This is an example of Excitability Hypothesis:
    hyp_x0 = hypo_builder.build_hypothesis_from_file(EP_NAME)

    # # This is an example of Mixed Hypothesis set manually by the user:
    # x0_indices = [hyp_x0.x0_indices[-1]]
    # x0_values = [hyp_x0.x0_values[-1]]
    # e_indices = hyp_x0.x0_indices[0:-1].tolist()
    # e_values = hyp_x0.x0_values[0:-1].tolist()
    # hyp_x0_E = hypo_builder.build_hypothesis_from_manual_input(e_values, e_indices, x0_values, x0_indices)

    # This is an example of x0_values mixed Excitability and Epileptogenicity Hypothesis set from file:
    all_regions_indices = np.array(range(head.number_of_regions))
    healthy_indices = np.delete(all_regions_indices, hyp_E.x0_indices + hyp_E.e_indices).tolist()
    hyp_x0_E = hypo_builder.build_hypothesis_from_file(EP_NAME, [16, 25])

    hypotheses = (hyp_x0, hyp_E, hyp_x0_E)

    # --------------------------Simulation preparations-----------------------------------
    # If you choose model...
    # Available models beyond the TVB Epileptor (they all encompass optional variations from the different papers):
    # EpileptorDP: similar to the TVB Epileptor + optional variations,
    # EpileptorDP2D: reduced 2D model, following Proix et all 2014 +optional variations,
    # EpleptorDPrealistic: starting from the TVB Epileptor + optional variations, but:
    #      -x0, Iext1, Iext2, slope and K become noisy state variables,
    #      -Iext2 and slope are coupled to z, g, or z*g in order for spikes to appear before seizure,
    #      -multiplicative correlated noise is also used
    # We don't want any time delays for the moment
    # head.connectivity.tract_lengths *= TIME_DELAYS_FLAG
    sim_builder = SimulatorBuilder(config.generic.MODE_TVB)
    if isequal_string(sim_type, "realistic"):
        sim_settings = sim_builder.set_model_name("EpileptorDPrealistic").set_simulated_period(50000).build_sim_settings()
        sim_settings.noise_type = COLORED_NOISE
        sim_settings.noise_ntau = 10
    elif isequal_string(sim_type, "fitting"):
        sim_builder.set_model_name("EpileptorDP2D").build_sim_settings()
        sim_settings = sim_builder
        sim_settings.noise_intensity = 1e-3
    elif isequal_string(sim_type, "paper"):
        sim_builder.set_model_name("Epileptor")
        sim_settings = sim_builder.build_sim_settings()
    else:
        sim_settings = sim_builder.build_sim_settings()

    # --------------------------Hypothesis and LSA-----------------------------------
    for hyp in hypotheses:
        logger.info("\n\nRunning hypothesis: " + hyp.name)
        logger.info("\n\nCreating model configuration...")
        builder = ModelConfigurationBuilder(hyp.number_of_regions)

        mcs_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_model_config_service.h5")
        writer.write_model_configuration_builder(builder, mcs_file)
        if test_write_read:
            logger.info("Written and read model configuration services are identical?: " +
                        str(assert_equal_objects(builder, reader.read_model_configuration_builder(mcs_file),
                                                 logger=logger)))

        if hyp.type == "Epileptogenicity":
            model_configuration = builder.build_model_from_E_hypothesis(hyp, head.connectivity.normalized_weights)
        else:
            model_configuration = builder.build_model_from_hypothesis(hyp, head.connectivity.normalized_weights)
        mc_path = os.path.join(config.out.FOLDER_RES, hyp.name + "_ModelConfig.h5")
        writer.write_model_configuration(model_configuration, mc_path)
        if test_write_read:
            logger.info("Written and read model configuration are identical?: " +
                        str(assert_equal_objects(model_configuration, reader.read_model_configuration(mc_path),
                                                 logger=logger)))
        # Plot nullclines and equilibria of model configuration
        plotter.plot_state_space(model_configuration, head.connectivity.region_labels,
                                 special_idx=hyp_x0.x0_indices + hyp_E.e_indices, model="6d", zmode="lin",
                                 figure_name=hyp.name + "_StateSpace")

        logger.info("\n\nRunning LSA...")
        lsa_service = LSAService(eigen_vectors_number=None, weighted_eigenvector_sum=True)
        lsa_hypothesis = lsa_service.run_lsa(hyp, model_configuration)

        lsa_path = os.path.join(config.out.FOLDER_RES, lsa_hypothesis.name + "_LSA.h5")
        lsa_config_path = os.path.join(config.out.FOLDER_RES, lsa_hypothesis.name + "_LSAConfig.h5")
        writer.write_hypothesis(lsa_hypothesis, lsa_path)
        writer.write_lsa_service(lsa_service, lsa_config_path)
        if test_write_read:
            logger.info("Written and read LSA services are identical?: " +
                        str(assert_equal_objects(lsa_service, reader.read_lsa_service(lsa_config_path), logger=logger)))
            logger.info("Written and read LSA hypotheses are identical (no input check)?: " +
                        str(assert_equal_objects(lsa_hypothesis, reader.read_hypothesis(lsa_path), logger=logger)))
        plotter.plot_lsa(lsa_hypothesis, model_configuration, lsa_service.weighted_eigenvector_sum,
                         lsa_service.eigen_vectors_number, head.connectivity.region_labels, None)

        if pse_flag:
            # --------------Parameter Search Exploration (PSE)-------------------------------
            logger.info("\n\nRunning PSE LSA...")
            pse_results = pse_from_lsa_hypothesis(lsa_hypothesis,
                                                  head.connectivity.normalized_weights,
                                                  head.connectivity.region_labels,
                                                  n_samples, param_range=0.1,
                                                  global_coupling=[{"indices": all_regions_indices}],
                                                  healthy_regions_parameters=[
                                                      {"name": "x0_values", "indices": healthy_indices}],
                                                  model_configuration_builder=builder,
                                                  lsa_service=lsa_service, logger=logger, save_flag=True)[0]
            plotter.plot_lsa(lsa_hypothesis, model_configuration, lsa_service.weighted_eigenvector_sum,
                             lsa_service.eigen_vectors_number, head.connectivity.region_labels, pse_results)

            pse_lsa_path = os.path.join(config.out.FOLDER_RES, lsa_hypothesis.name + "_PSE_LSA_results.h5")
            writer.write_dictionary(pse_results, pse_lsa_path)
            if test_write_read:
                logger.info("Written and read sensitivity analysis parameter search results are identical?: " +
                            str(assert_equal_objects(pse_results, reader.read_dictionary(pse_lsa_path), logger=logger)))

        if sa_pse_flag:
            # --------------Sensitivity Analysis Parameter Search Exploration (PSE)-------------------------------
            logger.info("\n\nrunning sensitivity analysis PSE LSA...")
            sa_results, pse_sa_results = \
                sensitivity_analysis_pse_from_lsa_hypothesis(lsa_hypothesis,
                                                             head.connectivity.normalized_weights,
                                                             head.connectivity.region_labels,
                                                             n_samples, method="sobol", param_range=0.1,
                                                             global_coupling=[{"indices": all_regions_indices,
                                                                               "bounds": [0.0, 2 *
                                                                                          builder.K_unscaled[
                                                                                              0]]}],
                                                             healthy_regions_parameters=[
                                                                 {"name": "x0_values", "indices": healthy_indices}],
                                                             model_configuration_builder=builder,
                                                             lsa_service=lsa_service, logger=logger)
            plotter.plot_lsa(lsa_hypothesis, model_configuration, lsa_service.weighted_eigenvector_sum,
                             lsa_service.eigen_vectors_number, head.connectivity.region_labels, pse_sa_results,
                             title="SA PSE Hypothesis Overview")

            sa_pse_path = os.path.join(config.out.FOLDER_RES, lsa_hypothesis.name + "_SA_PSE_LSA_results.h5")
            sa_lsa_path = os.path.join(config.out.FOLDER_RES, lsa_hypothesis.name + "_SA_LSA_results.h5")
            writer.write_dictionary(pse_sa_results, sa_pse_path)
            writer.write_dictionary(sa_results, sa_lsa_path)
            if test_write_read:
                logger.info("Written and read sensitivity analysis results are identical?: " +
                            str(assert_equal_objects(sa_results, reader.read_dictionary(sa_lsa_path), logger=logger)))
                logger.info("Written and read sensitivity analysis parameter search results are identical?: " +
                            str(assert_equal_objects(pse_sa_results, reader.read_dictionary(sa_pse_path),
                                                     logger=logger)))

        if sim_flag:
            # ------------------------------Simulation--------------------------------------
            logger.info("\n\nConfiguring simulation from model_configuration...")
            model = sim_builder.generate_model(model_configuration)
            if isequal_string(sim_type, "realistic"):
                model.tau0 = 30000.0
                model.tau1 = 0.2
                model.slope = 0.25
            elif isequal_string(sim_type, "fitting"):
                model.tau0 = 10.0
                model.tau1 = 0.5
            sim, sim_settings, model = sim_builder.build_simulator_TVB_from_model_sim_settings(model_configuration,
                                                                                 head.connectivity, model, sim_settings)

            # Integrator and initial conditions initialization.
            # By default initial condition is set right on the equilibrium point.
            writer.write_generic(sim.model, config.out.FOLDER_RES, lsa_hypothesis.name + "_sim_model.h5")
            logger.info("\n\nSimulating...")
            ttavg, tavg_data, status = sim.launch_simulation(report_every_n_monitor_steps=100)

            sim_path = os.path.join(config.out.FOLDER_RES, lsa_hypothesis.name + "_sim_settings.h5")
            writer.write_simulation_settings(sim.simulation_settings, sim_path)
            if test_write_read:
                # TODO: find out why it cannot set monitor expressions
                logger.info("Written and read simulation settings are identical?: " +
                            str(assert_equal_objects(sim.simulation_settings,
                                                     reader.read_simulation_settings(sim_path), logger=logger)))
            if not status:
                logger.warning("\nSimulation failed!")
            else:
                time = np.array(ttavg, dtype='float32')
                output_sampling_time = np.mean(np.diff(time))
                tavg_data = tavg_data[:, :, :, 0]
                logger.info("\n\nSimulated signal return shape: %s", tavg_data.shape)
                logger.info("Time: %s - %s", time[0], time[-1])
                logger.info("Values: %s - %s", tavg_data.min(), tavg_data.max())
                # Variables of interest in a dictionary:
                vois_ts_dict = prepare_vois_ts_dict(sim_settings.monitor_expressions, tavg_data)
                vois_ts_dict['time'] = time
                vois_ts_dict['time_units'] = 'msec'
                vois_ts_dict = compute_seeg_and_write_ts_h5_file(config.out.FOLDER_RES, lsa_hypothesis.name + "_ts.h5",
                                                                 sim.model, vois_ts_dict, output_sampling_time,
                                                                 sim_settings.simulated_period,
                                                                 hpf_flag=True, hpf_low=10.0, hpf_high=512.0,
                                                                 sensors_list=head.sensorsSEEG)
                # Plot results
                if model._ui_name is "EpileptorDP2D":
                    spectral_raster_plot = False
                    trajectories_plot = True
                else:
                    spectral_raster_plot = "lfp"
                    trajectories_plot = False
                plotter.plot_sim_results(sim.model, lsa_hypothesis.lsa_propagation_indices, vois_ts_dict,
                                         head.sensorsSEEG, hpf_flag=True, trajectories_plot=trajectories_plot,
                                         spectral_raster_plot=spectral_raster_plot, log_scale=True)
                # Optionally save results in mat files
                # from scipy.io import savemat
                # savemat(os.path.join(FOLDER_RES, lsa_hypothesis.name + "_ts.mat"), vois_ts_dict)


if __name__ == "__main__":
    head_folder = os.path.join(os.path.expanduser("~"),
                               'Dropbox', 'Work', 'VBtech', 'VEP', "results", "CC", "TVB3", "Head")
    output = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "tests")
    config = Config(head_folder=head_folder, output_base=output, separate_by_run=True)
    main_vep(config)
    main_vep()
