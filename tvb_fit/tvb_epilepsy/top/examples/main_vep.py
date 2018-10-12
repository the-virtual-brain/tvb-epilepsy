"""
Entry point for working with VEP
"""
import os
import numpy as np

from tvb_fit.base.utils.data_structures_utils import assert_equal_objects, isequal_string, ensure_list
from tvb_fit.base.utils.log_error_utils import initialize_logger
from tvb_fit.io.tvb_data_reader import TVBReader
from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.constants.model_constants import K_UNSCALED_DEF, TAU0_DEF, TAU1_DEF
from tvb_fit.base.constants import COLORED_NOISE
from tvb_fit.tvb_epilepsy.base.constants.model_constants import PMODE_DEF
from tvb_fit.tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_fit.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_fit.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_fit.tvb_epilepsy.service.simulator.simulator_builder import SimulatorBuilder
from tvb_fit.tvb_epilepsy.top.scripts.pse_scripts import pse_from_lsa_hypothesis
from tvb_fit.tvb_epilepsy.top.scripts.sensitivity_analysis_sripts import sensitivity_analysis_pse_from_lsa_hypothesis
from tvb_fit.tvb_epilepsy.top.scripts.simulation_scripts import compute_seeg_and_write_ts_to_h5
from tvb_fit.tvb_epilepsy.service.lsa_service import LSAService
from tvb_fit.tvb_epilepsy.io.h5_reader import H5Reader
from tvb_fit.tvb_epilepsy.io.h5_writer import H5Writer
from tvb_fit.tvb_epilepsy.plot.plotter import Plotter

PSE_FLAG = True
SA_PSE_FLAG = False
SIM_FLAG = True
EP_NAME = "clinical_hypothesis_preseeg"


def main_vep(config=Config(), ep_name=EP_NAME, K_unscaled=K_UNSCALED_DEF, ep_indices=[], hyp_norm=0.99, manual_hypos=[],
             sim_type="paper", pse_flag=PSE_FLAG, sa_pse_flag=SA_PSE_FLAG, sim_flag=SIM_FLAG, n_samples=100,
             test_write_read=False):
    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
    # -------------------------------Reading data-----------------------------------
    reader = TVBReader() if config.input.IS_TVB_MODE else H5Reader()
    writer = H5Writer()
    plotter = Plotter(config)
    logger.info("Reading from: " + config.input.HEAD)
    head = reader.read_head(config.input.HEAD)
    plotter.plot_head(head)
    if test_write_read:
        writer.write_head(head, os.path.join(config.out.FOLDER_RES, "Head"))
    # --------------------------Hypothesis definition-----------------------------------

    hypotheses = []
    # Reading a h5 file:

    if len(ep_name) > 0:
        # For an Excitability Hypothesis you leave e_indices empty
        # For a Mixed Hypothesis: you give as e_indices some indices for values > 0
        # For an Epileptogenicity Hypothesis: you give as e_indices all indices for values > 0
        hyp_file = HypothesisBuilder(head.connectivity.number_of_regions, config=config).set_normalize(hyp_norm). \
            build_hypothesis_from_file(ep_name, e_indices=ep_indices)
        hyp_file.name += ep_name
        # print(hyp_file.string_regions_disease(head.connectivity.region_labels))
        hypotheses.append(hyp_file)

    hypotheses += manual_hypos

    # --------------------------Hypothesis and LSA-----------------------------------
    for hyp in hypotheses:
        logger.info("\n\nRunning hypothesis: " + hyp.name)

        all_regions_indices = np.array(range(head.number_of_regions))
        healthy_indices = np.delete(all_regions_indices, hyp.regions_disease_indices).tolist()

        logger.info("\n\nCreating model configuration...")
        model_config_builder = ModelConfigurationBuilder("EpileptorDP2D", head.connectivity, K_unscaled=K_unscaled). \
                                    set_parameter("tau1", TAU1_DEF).set_parameter("tau0", TAU0_DEF)
        mcs_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_model_config_builder.h5")
        writer.write_model_configuration_builder(model_config_builder, mcs_file)
        if test_write_read:
            logger.info("Written and read model configuration builders are identical?: " +
                        str(assert_equal_objects(model_config_builder,
                                                 reader.read_model_configuration_builder(mcs_file), logger=logger)))
        # Fix healthy regions to default equilibria:
        # model_configuration = \
        #        model_config_builder.build_model_from_E_hypothesis(hyp)
        # Fix healthy regions to default x0s:
        model_configuration = model_config_builder.build_model_from_hypothesis(hyp)
        mc_path = os.path.join(config.out.FOLDER_RES, hyp.name + "_ModelConfig.h5")
        writer.write_model_configuration(model_configuration, mc_path)
        if test_write_read:
            logger.info("Written and read model configuration are identical?: " +
                        str(assert_equal_objects(model_configuration,
                                                 reader.read_model_configuration(mc_path),
                                                 logger=logger)))
        # Plot nullclines and equilibria of model configuration
        plotter.plot_state_space(model_configuration, head.connectivity.region_labels,
                                 special_idx=hyp.regions_disease_indices,
                                 figure_name=hyp.name + "_StateSpace")

        logger.info("\n\nRunning LSA...")
        lsa_service = LSAService(eigen_vectors_number=1)
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
                             lsa_service.eigen_vectors_number, head.connectivity.region_labels, None,
                             lsa_service=lsa_service)

        if pse_flag:
            # --------------Parameter Search Exploration (PSE)-------------------------------
            logger.info("\n\nRunning PSE LSA...")
            pse_results = pse_from_lsa_hypothesis(n_samples, lsa_hypothesis,
                                                  model_configuration.connectivity,
                                                  model_config_builder, lsa_service,
                                                  head.connectivity.region_labels,
                                                  param_range=0.1,
                                                  global_coupling=[{"indices": all_regions_indices}],
                                                  healthy_regions_parameters=[
                                                      {"name": "x0_values", "indices": healthy_indices}],
                                                  logger=logger, save_flag=True)[0]
            plotter.plot_lsa(lsa_hypothesis, model_configuration, lsa_service.weighted_eigenvector_sum,
                             lsa_service.eigen_vectors_number, head.connectivity.region_labels, pse_results)

            pse_lsa_path = os.path.join(config.out.FOLDER_RES, lsa_hypothesis.name + "_PSE_LSA_results.h5")
            writer.write_dictionary(pse_results, pse_lsa_path)
            if test_write_read:
                logger.info("Written and read parameter search results are identical?: " +
                            str(assert_equal_objects(pse_results, reader.read_dictionary(pse_lsa_path), logger=logger)))

        if sa_pse_flag:
            # --------------Sensitivity Analysis Parameter Search Exploration (PSE)-------------------------------
            logger.info("\n\nrunning sensitivity analysis PSE LSA...")
            sa_results, pse_sa_results = \
                sensitivity_analysis_pse_from_lsa_hypothesis(n_samples, lsa_hypothesis,
                                                             model_configuration.connectivity,
                                                             model_config_builder, lsa_service,
                                                             head.connectivity.region_labels,
                                                             method="sobol", param_range=0.1,
                                                             global_coupling=[{"indices": all_regions_indices,
                                                                               "bounds": [0.0, 2 *
                                                                                          model_config_builder.K_unscaled[
                                                                                              0]]}],
                                                             healthy_regions_parameters=[
                                                                 {"name": "x0_values", "indices": healthy_indices}],
                                                             config=config)
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
            # --------------------------Simulation preparations-----------------------------------
            # If you choose model...
            # Available models beyond the TVB Epileptor (they all encompass optional variations from the different papers):
            # EpileptorDP: similar to the TVB Epileptor + optional variations,
            # EpileptorDP2D: reduced 2D model, following Proix et all 2014 +optional variations,
            # EpleptorDPrealistic: starting from the TVB Epileptor + optional variations, but:
            #      -x0, Iext1, Iext2, slope and K become noisy state variables,
            #      -Iext2 and slope are coupled to z, g, or z*g in order for spikes to appear before seizure,
            #      -correlated noise is also used
            # We don't want any time delays for the moment
            head.connectivity.tract_lengths *= config.simulator.USE_TIME_DELAYS_FLAG
            sim_types = ensure_list(sim_type)
            for sim_type in sim_types:
                # ------------------------------Simulation--------------------------------------
                logger.info("\n\nConfiguring %s simulation from model_configuration..." % sim_type)
                if isequal_string(sim_type, "realistic"):
                    sim_builder = SimulatorBuilder(model_configuration).set_model("EpileptorDPrealistic"). \
                        set_fs(2048.0).set_simulation_length(60000.0)
                    sim_builder.model_config.tau0 = 60000.0
                    sim_builder.model_config.tau1 = 0.2
                    sim_builder.model_config.slope = 0.25
                    sim_builder.model_config.pmode = np.array([PMODE_DEF])
                    sim_settings = sim_builder.build_sim_settings()
                    sim_settings.noise_type = COLORED_NOISE
                    sim_settings.noise_ntau = 20
                    # Necessary a more stable integrator:
                    sim_settings.integrator_type = "Dop853Stochastic"
                elif isequal_string(sim_type, "fitting"):
                    sim_builder = SimulatorBuilder(model_configuration).set_model("EpileptorDP2D"). \
                        set_fs(2048.0).set_fs_monitor(2048.0).set_simulation_length(300.0)
                    sim_builder.model_config.tau0 = 30.0
                    sim_builder.model_config.tau1 = 0.5
                    sim_settings = sim_builder.build_sim_settings()
                    sim_settings.noise_intensity = np.array([0.0, 1e-5])
                elif isequal_string(sim_type, "reduced"):
                    sim_builder = \
                        SimulatorBuilder(model_configuration).set_model("EpileptorDP2D").set_fs(
                            4096.0).set_simulation_length(1000.0)
                    sim_settings = sim_builder.build_sim_settings()
                elif isequal_string(sim_type, "paper"):
                    sim_builder = SimulatorBuilder(model_configuration).set_model("Epileptor")
                    sim_settings = sim_builder.build_sim_settings()
                else:
                    sim_builder = SimulatorBuilder(model_configuration).set_model("EpileptorDP")
                    sim_settings = sim_builder.build_sim_settings()

                # Integrator and initial conditions initialization.
                # By default initial condition is set right on the equilibrium point.
                sim, sim_settings = \
                    sim_builder.build_simulator_TVB_from_model_sim_settings(head.connectivity, sim_settings)
                sim_path = os.path.join(config.out.FOLDER_RES, lsa_hypothesis.name + "_"
                                        + sim_type + "_sim_settings.h5")
                model_path = os.path.join(config.out.FOLDER_RES,
                                          lsa_hypothesis.name + sim_type + "_model.h5")
                writer.write_simulation_settings(sim.settings, sim_path)
                writer.write_simulator_model(sim.model, model_path, sim.connectivity.number_of_regions)
                if test_write_read:
                    # TODO: find out why it cannot set monitor expressions
                    logger.info("Written and read simulation settings are identical?: " +
                                str(assert_equal_objects(sim.settings,
                                                         reader.read_simulation_settings(sim_path), logger=logger)))
                    # logger.info("Written and read simulation model are identical?: " +
                    #             str(assert_equal_objects(sim.model,
                    #                                      reader.read_epileptor_model(model_path), logger=logger)))

                logger.info("\n\nSimulating %s..." % sim_type)
                sim_output, status = sim.launch_simulation(report_every_n_monitor_steps=100, timeseries=Timeseries)
                if not status:
                    logger.warning("\nSimulation failed!")
                else:
                    time = np.array(sim_output.time_line).astype("f")
                    logger.info("\n\nSimulated signal return shape: %s", sim_output.shape)
                    logger.info("Time: %s - %s", time[0], time[-1])
                    logger.info("Values: %s - %s", sim_output.data.min(), sim_output.data.max())
                    if not status:
                        logger.warning("\nSimulation failed!")
                    else:
                        sim_output, seeg = compute_seeg_and_write_ts_to_h5(sim_output, sim.model, head.sensorsSEEG,
                                                                           os.path.join(config.out.FOLDER_RES,
                                                                                        sim_type + "_ts.h5"),
                                                                           seeg_gain_mode="lin", hpf_flag=True,
                                                                           hpf_low=10.0, hpf_high=512.0)

                        # Plot results
                        plotter.plot_simulated_timeseries(sim_output, sim.model, lsa_hypothesis.lsa_propagation_indices,
                                                          seeg_dict=seeg, spectral_raster_plot=False,
                                                          title_prefix=hyp.name, spectral_options={"log_scale": True})


if __name__ == "__main__":
    head_folder = os.path.join(os.path.expanduser("~"),
                               'Dropbox', 'Work', 'VBtech', 'VEP', "results", "CC", "TVB3", "Head")
    output = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "tests")
    config = Config(head_folder=head_folder, output_base=output, separate_by_run=False)
    main_vep(config, "clinical_hypothesis_postseeg", ep_indices=[1, 26],
             sim_type=[ "paper", "default", "fitting", "reduced"], test_write_read=True)  #",, "realistic"
    # main_vep()
