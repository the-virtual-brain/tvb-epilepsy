# encoding=utf8

import os

import numpy as np

from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.constants.model_constants import K_DEF, TAU1_DEF, TAU0_DEF
from tvb_epilepsy.base.constants.model_inversion_constants import XModes, SDE_MODES, PriorsModes, TARGET_DATA_TYPE, \
    OBSERVATION_MODELS, BIPOLAR, TARGET_DATA_PREPROCESSING
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.service.model_inversion.probabilistic_models_builders import SDEProbabilisticModelBuilder
from tvb_epilepsy.service.model_inversion.model_inversion_services import SDEModelInversionService
from tvb_epilepsy.service.model_inversion.stan.cmdstan_service import CmdStanService
from tvb_epilepsy.service.model_inversion.vep_stan_dict_builder import build_stan_model_data_dict
from tvb_epilepsy.top.scripts.simulation_scripts import from_model_configuration_to_simulation
from tvb_epilepsy.top.scripts.fitting_scripts import set_model_config_LSA, set_empirical_data, set_simulated_target_data
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.io.h5_writer import H5Writer


def set_hypotheses(head, config):
    # Formulate a VEP hypothesis manually
    hyp_builder = HypothesisBuilder(head.connectivity.number_of_regions, config)  # .set_normalize(0.99)

    # Regions of Pathological Excitability hypothesis:
    x0_indices = [2, 24]
    x0_values = [0.01, 0.01]
    hyp_builder.set_x0_hypothesis(x0_indices, x0_values)

    # Regions of Model Epileptogenicity hypothesis:
    e_indices = [1, 26]
    # e_indices = list(range(head.connectivity.number_of_regions))
    # e_indices.remove(2)
    # e_indices.remove(25)
    # e_values = np.zeros((head.connectivity.number_of_regions,)) + 0.01
    # e_values[[1, 26]] = 0.99
    # e_values = np.delete(e_values, [2, 25]).tolist()
    e_values = np.array([1.5, 1.25])  # np.array([0.99] * 2)
    hyp_builder.set_e_hypothesis(e_indices, e_values)

    # Regions of Connectivity hypothesis:
    # w_indices = []  # [(0, 1), (0, 2)]
    # w_values = []  # [0.5, 2.0]
    # hypo_builder.set_w_indices(w_indices).set_w_values(w_values)

    hypothesis1 = hyp_builder.build_hypothesis()

    e_indices = [1, 26]  # [1, 2, 25, 26]
    hypothesis2 = hyp_builder.build_hypothesis_from_file("clinical_hypothesis_postseeg", e_indices)
    # Change something manually if necessary
    # hypothesis2.x0_values = [0.01, 0.01]

    return (hypothesis1, hypothesis2)


def main_fit_sim_hyplsa(stan_model_name="vep_sde", empirical_file="",
                        observation_model=OBSERVATION_MODELS.SEEG_LOGPOWER.value, sensors_lbls=[], sensor_id=0,
                        times_on_off=[], sim_times_on_off=[80.0, 120.0], preprocessing_sequence=TARGET_DATA_PREPROCESSING,
                        fitmethod="optimizing", pse_flag=True, fit_flag=True, config=Config(), **kwargs):

    def path(name):
        if len(name) > 0:
            return base_path + "_" + name + ".h5"
        else:
            return base_path + ".h5"

    # Prepare necessary services:
    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
    reader = H5Reader()
    writer = H5Writer()
    plotter = Plotter(config)

    # Read head
    logger.info("Reading from: " + config.input.HEAD)
    head = reader.read_head(config.input.HEAD)
    sensors = head.get_sensors_id(sensor_ids=sensor_id)
    plotter.plot_head(head)

    # Set hypotheses:
    hypotheses = set_hypotheses(head, config)

    # ------------------------------Stan model and service--------------------------------------
    model_code_path = os.path.join(config.generic.PROBLSTC_MODELS_PATH, stan_model_name + ".stan")
    stan_service = CmdStanService(model_name=stan_model_name, model_code_path=model_code_path, fitmethod=fitmethod,
                                      config=config)
    stan_service.set_or_compile_model()

    for hyp in hypotheses[:1]:
        base_path = os.path.join(config.out.FOLDER_RES, hyp.name)
        # Set model configuration and compute LSA
        model_configuration, lsa_hypothesis, pse_results = \
            set_model_config_LSA(head, hyp, reader, config, K_unscaled=3*K_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF,
                                 pse_flag=pse_flag, plotter=plotter, writer=writer)

        # -------------------------- Get model_data and observation signals: -------------------------------------------
        # Create model inversion service (stateless)
        problstc_model_file = path("ProblstcModel")
        model_data_file = path("ModelData")
        target_data_file = path("TargetData")
        if os.path.isfile(problstc_model_file) and os.path.isfile(model_data_file) and os.path.isfile(target_data_file):
            # Read existing probabilistic model and model data...
            probabilistic_model = reader.read_probabilistic_model(problstc_model_file)
            model_data = stan_service.load_model_data_from_file(model_data_path=model_data_file)
            target_data = reader.read_timeseries(target_data_file)
        else:
            model_inversion = SDEModelInversionService()

            # ...or generate a new probabilistic model and model data
            probabilistic_model = \
                SDEProbabilisticModelBuilder(model_name="vep_sde_ins.stan", model_config=model_configuration,
                                             xmode=XModes.X0MODE.value, priors_mode=PriorsModes.NONINFORMATIVE.value,
                                             sde_mode=SDE_MODES.NONCENTERED.value, observation_model=observation_model,
                                             K=model_configuration.K).generate_model(generate_parameters=False)

            # Update active model's active region nodes
            e_values = pse_results.get("e_values_mean", model_configuration.e_values)
            lsa_propagation_strength = pse_results.get("lsa_propagation_strengths_mean",
                                                           lsa_hypothesis.lsa_propagation_strengths)
            model_inversion.active_e_th = 0.2
            probabilistic_model = \
                model_inversion.update_active_regions(probabilistic_model, e_values=e_values,
                                                      lsa_propagation_strengths=lsa_propagation_strength, reset=True)

            # --------------------- Get prototypical simulated data (simulate if necessary) ----------------------------
            source_ts = from_model_configuration_to_simulation(model_configuration, head, lsa_hypothesis,
                                                               sim_type="fitting", ts_file=path("ts"),
                                                               config=config, plotter=plotter)[0]["source"]. \
                        get_time_window_by_units(sim_times_on_off[0], sim_times_on_off[1])


            # Now some scripts for settting and preprocessing target signals:
            if os.path.isfile(empirical_file):
                probabilistic_model.target_data_type = TARGET_DATA_TYPE.EMPIRICAL.value
                # -------------------------- Get empirical data (preprocess edf if necessary) --------------------------
                signals = set_empirical_data(empirical_file, path("ts_empirical"),
                                             head, sensors_lbls, sensor_id, probabilistic_model.time_length,
                                             times_on_off=times_on_off, label_strip_fun=lambda s: s.split("POL ")[-1],
                                             preprocessing=preprocessing_sequence, plotter=plotter, title_prefix=hyp.name)
            else:
                # --------------------- Get fitting target simulated data (simulate if necessary) ----------------------
                probabilistic_model.target_data_type = TARGET_DATA_TYPE.SYNTHETIC.value
                signals, simulator = \
                    set_simulated_target_data(path("ts_fit"), model_configuration, head, lsa_hypothesis,
                                              probabilistic_model,
                                              sensor_id, sim_type="fitting", times_on_off=times_on_off,
                                              config=config,
                                              # Maybe change some of those for Epileptor 6D simulations:
                                              bipolar=False, preprocessing=preprocessing_sequence,
                                              plotter=plotter, title_prefix=hyp.name)

            # -------------------------- Select and set target data from signals ---------------------------------------
            if probabilistic_model.observation_model in OBSERVATION_MODELS.SEEG.value:
                model_inversion.auto_selection = "correlation-power"
                model_inversion.sensors_per_electrode = 2
            target_data, probabilistic_model, gain_matrix = \
                model_inversion.set_target_data_and_time(signals, probabilistic_model, head=head, sensors=sensors)

            plotter.plot_raster({'Target Signals': target_data.squeezed}, target_data.time_line,
                                time_units=target_data.time_unit, title=hyp.name + ' Target Signals raster',
                                offset=0.1, labels=target_data.space_labels)
            plotter.plot_timeseries({'Target Signals': target_data.squeezed}, target_data.time_line,
                                    time_units=target_data.time_unit,
                                    title=hyp.name + ' Target Signals', labels=target_data.space_labels)
            writer.write_timeseries(target_data, target_data_file)

            #---------------------------------Finally set priors for the parameters-------------------------------------
            probabilistic_model.parameters.update(
                SDEProbabilisticModelBuilder(probabilistic_model). \
                    generate_parameters([XModes.X0MODE.value, "sigma_"+XModes.X0MODE.value,
                                         "x1_init", "z_init", "tau1",  # "tau0", "K", "x1",
                                         "sigma", "dZt", "epsilon", "scale", "offset"],
                                        target_data, source_ts, gain_matrix))
            plotter.plot_probabilistic_model(probabilistic_model, hyp.name + " Probabilistic Model")
            writer.\
              write_probabilistic_model(probabilistic_model, model_configuration.number_of_regions, problstc_model_file)

            # Construct the stan model data dict:
            model_data = build_stan_model_data_dict(probabilistic_model, target_data.squeezed,
                                                    model_configuration.model_connectivity, gain_matrix,
                                                    time=target_data.time_line)
            # # ...or interface with INS stan models
            # from tvb_epilepsy.service.model_inversion.vep_stan_dict_builder import \
            #   build_stan_model_data_dict_to_interface_ins
            # model_data = build_stan_model_data_dict_to_interface_ins(probabilistic_model, target_data.squeezed,
            #                                                          model_configuration.model_connectivity, gain_matrix,
            #                                                          time=target_data.time_line)
            writer.write_dictionary(model_data, model_data_file)

        # -------------------------- Fit and get estimates: ------------------------------------------------------------
        n_chains_or_runs = 2
        output_samples = max(int(np.round(1000.0 / n_chains_or_runs)), 500)
        # Sampling (HMC)
        num_samples = output_samples
        num_warmup = 1000
        max_depth = 12
        delta = 0.9
        # ADVI or optimization:
        iter = 1000000
        tol_rel_obj = 1e-6
        if fitmethod.find("sampl") >= 0:
            skip_samples = num_warmup
        else:
            skip_samples = 0
        prob_model_name = probabilistic_model.name.split(".")[0]
        if fit_flag:
            estimates, samples, summary = stan_service.fit(debug=0, simulate=0, model_data=model_data, refresh=1,
                                                           n_chains_or_runs=n_chains_or_runs,
                                                           iter=iter, tol_rel_obj=tol_rel_obj,
                                                           num_warmup=num_warmup, num_samples=num_samples,
                                                           max_depth=max_depth, delta=delta,
                                                           save_warmup=1, plot_warmup=1, **kwargs)
            writer.write_generic(estimates, path(prob_model_name + "_FitEst"))
            writer.write_generic(samples, path(prob_model_name + "_FitSamples"))
            if summary is not None:
                writer.write_generic(summary, path(prob_model_name + "_FitSummary"))
        else:
            estimates, samples, summary = stan_service.read_output()
            if fitmethod.find("sampl") >= 0:
                plotter.plot_HMC(samples, figure_name=hyp.name + "-" + prob_model_name + " HMC NUTS trace")

        # Model comparison:
        # scale_signal, offset_signal, time_scale, epsilon, sigma -> 5 (+ K = 6)
        # x0[active] -> probabilistic_model.model.number_of_active_regions
        # x1init[active], zinit[active] -> 2 * probabilistic_model.number_of_active_regions
        # dZt[active, t] -> probabilistic_model.number_of_active_regions * (probabilistic_model.time_length-1)
        number_of_total_params =\
            5 + probabilistic_model.number_of_active_regions * (3 + (probabilistic_model.time_length-1))
        info_crit = \
            stan_service.compute_information_criteria(samples, number_of_total_params, skip_samples=skip_samples,
                                                      # parameters=["amplitude_star", "offset_star", "epsilon_star",
                                                      #                  "sigma_star", "time_scale_star", "x0_star",
                                                      #                  "x_init_star", "z_init_star", "z_eta_star"],
                                                      merge_chains_or_runs_flag=False)

        writer.write_generic(info_crit, path(prob_model_name + "_InfoCrit"))

        Rhat = stan_service.get_Rhat(summary)
        # Interface backwards with INS stan models
        # from tvb_epilepsy.service.model_inversion.vep_stan_dict_builder import convert_params_names_from_ins
        # estimates, samples, Rhat, model_data = \
        #     convert_params_names_from_ins([estimates, samples, Rhat, model_data])
        if fitmethod.find("opt") < 0 and Rhat is not None:
            stats = {"Rhat": Rhat}
        else:
            stats = None

        # -------------------------- Plot fitting results: ------------------------------------------------------------
        # if stan_service.fitmethod.find("opt") < 0:
        plotter.plot_fit_results(estimates, samples, model_data, target_data, probabilistic_model, info_crit,
                                 stats=stats,
                                 pair_plot_params=["tau1", "tau0", "K", "sigma", "epsilon", "scale", "offset"],  #
                                 region_violin_params=["x0", "x1_init", "z_init"],
                                 regions_labels=head.connectivity.region_labels, skip_samples=skip_samples,
                                 title_prefix=hyp.name + "-" + prob_model_name)


        # -------------------------- Reconfigure model after fitting:---------------------------------------------------
        for id_est, est in enumerate(ensure_list(estimates)):
            K = est.get("K", model_configuration.K)
            tau1 = est.get("tau1", model_configuration.tau1)
            tau0 = est.get("tau0", model_configuration.tau0)
            fit_model_configuration_builder = \
                ModelConfigurationBuilder(hyp.number_of_regions, K=K * hyp.number_of_regions, tau1=tau1, tau0=tau0)
            x0_values_fit = model_configuration.x0_values
            x0_values_fit[probabilistic_model.active_regions] = \
                fit_model_configuration_builder._compute_x0_values_from_x0_model(est['x0'])
            hyp_fit = HypothesisBuilder().set_nr_of_regions(head.connectivity.number_of_regions).\
                                          set_name('fit' + str(id_est+1) + "_" + hyp.name).\
                                          set_x0_hypothesis(list(probabilistic_model.active_regions),
                                                            x0_values_fit[probabilistic_model.active_regions]).\
                                          build_hypothesis()
            base_path = os.path.join(config.out.FOLDER_RES, hyp_fit.name)
            writer.write_hypothesis(hyp_fit, path(""))

            model_configuration_fit = \
                fit_model_configuration_builder.build_model_from_hypothesis(hyp_fit,  # est["MC"]
                                                                            model_configuration.model_connectivity)

            writer.write_model_configuration(model_configuration_fit, path("ModelConfig"))

            # Plot nullclines and equilibria of model configuration
            plotter.plot_state_space(model_configuration_fit, region_labels=head.connectivity.region_labels,
                                     special_idx=probabilistic_model.active_regions, model="6d", zmode="lin",
                                     figure_name=hyp_fit.name + "_Nullclines and equilibria")
        logger.info("Done!")


if __name__ == "__main__":

    user_home = os.path.expanduser("~")
    head_folder = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "CC", "TVB3", "Head")
    SEEG_data = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', "data/CC", "TVB3",
                             "raw/seeg/ts_seizure")

    if user_home == "/home/denis":
        output = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results",
                              "INScluster/fit_tau030_Kfixed/empirical_noninfo")
        config = Config(head_folder=head_folder, raw_data_folder=SEEG_data, output_base=output, separate_by_run=False)
        config.generic.C_COMPILER = "g++"
        config.generic.CMDSTAN_PATH = "/soft/stan/cmdstan-2.17.0"

    elif user_home == "/Users/lia.domide":
        config = Config(head_folder="/WORK/episense/tvb-epilepsy/data/TVB3/Head",
                        raw_data_folder="/WORK/episense/tvb-epilepsy/data/TVB3/ts_seizure")
        config.generic.CMDSTAN_PATH = "/WORK/episense/cmdstan-2.17.1"

    else:
        output = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "fit_test")
        config = Config(head_folder=head_folder, raw_data_folder=SEEG_data, output_base=output, separate_by_run=False)
        config.generic.CMDSTAN_PATH = config.generic.CMDSTAN_PATH + "_precompiled"

    # TVB3 larger preselection:
    sensors_lbls = \
        [u"B'1", u"B'2", u"B'3", u"B'4",
         u"F'1", u"F'2", u"F'3", u"F'4", u"F'5", u"F'6", u"F'7", u"F'8", u"F'9", u"F'10", u"F'11",
         u"G'1", u"G'2", u"G'3", u"G'4", u"G'8", u"G'9", u"G'10", u"G'11", u"G'12", u"G'13", u"G'14", u"G'15",
         u"L'1", u"L'2", u"L'3", u"L'4", u"L'5", u"L'6", u"L'7", u"L'8", u"L'9", u"L'10", u"L'11", u"L'12", u"L'13",
         u"M'1", u"M'2", u"M'3", u"M'7", u"M'8", u"M'9", u"M'10", u"M'11", u"M'12", u"M'13", u"M'14", u"M'15",
         u"O'1", u"O'2", u"O'3", u"O'6", u"O'7", u"O'8", u"O'9", u"O'10", u"O'11", u"O'12", # u"O'13"
         u"P'1", u"P'2", u"P'3", u"P'8", u"P'10", u"P'11", u"P'12", u"P'13", u"P'14", u"P'15", u"P'16",
         u"R'1", u"R'2", u"R'3", u"R'4", u"R'7", u"R'8", u"R'9",
         ]
    sensors_inds = [0, 1, 2, 3,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                    28, 29, 30, 31, 36, 37, 38, 39, 40, 41, 42,
                    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                    58, 59, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                    74, 75, 76, 79, 80, 81, 82, 83, 84, 85, # 86,
                    90, 91, 92, 97, 99, 100, 101, 102, 103, 104, 105,
                    106, 107, 108, 109, 112, 113, 114
                    ]
    # TVB3 preselection:
    # sensors_lbls = [u"G'1", u"G'2", u"G'3", u"G'8", u"G'9", u"G'10", u"G'11", u"G'12", u"M'6", u"M'7", u"M'8", u"L'4",
    #                 u"L'5",  u"L'6", u"L'7", u"L'8", u"L'9"]
    # sensors_inds = [28, 29, 30, 35, 36, 37, 38, 39, 63, 64, 65, 47, 48, 49, 50, 51, 52]
    # TVB3 selection:
    # sensors_lbls = [u"G'1", u"G'2", u"G'11", u"G'12", u"M'7", u"M'8", u"L'5", u"L'6"]
    # sensors_inds = [28, 29, 38, 39, 64, 65, 48, 49]
    # Simulation times_on_off
    sim_times_on_off = [80.0, 120.0]  # for "fitting" simulations with tau0=30.0
    EMPIRICAL = True
    if EMPIRICAL:
        seizure = 'SZ1_0001.edf'
        times_on_off = (np.array([15.0, 30.0]) * 1000.0).tolist() #(np.array([15.0, 35.0]) * 1000.0).tolist()
        # sensors_filename = "SensorsSEEG_116.h5"
        # # TVB4 preselection:
        # sensors_lbls = [u"D5", u"D6", u"D7",  u"D8", u"D9", u"D10", u"Z9", u"Z10", u"Z11", u"Z12", u"Z13", u"Z14",
        #                 u"S1", u"S2", u"S3", u"D'3", u"D'4", u"D'10", u"D'11", u"D'12", u"D'13", u"D'14"]
        # sensors_inds = [4, 5, 6, 7, 8, 9, 86, 87, 88, 89, 90, 91, 94, 95, 96, 112, 113, 119, 120, 121, 122, 123]
        # # TVB4:
        # seizure = 'SZ3_0001.edf'
        # sensors_filename = "SensorsSEEG_210.h5"
        # times_on_off = [20.0, 100.0]
    else:
        times_on_off = sim_times_on_off # for "fitting" simulations with tau0=30.0
        # times_on_off = [50.0, 550.0]  # for "paper" simulations
        # times_on_off = [1100.0, 1300.0]  # for "fitting" simulations with tau0=300.0
    stan_model_name = "vep_sde"
    fitmethod = "sample"  # "sample"  # "advi" or "opt"
    observation_model = OBSERVATION_MODELS.SEEG_LOGPOWER.value  # OBSERVATION_MODELS.SOURCE_POWER.value  #
    preprocessing = ["filter", "abs", "convolve"]
    log_flag = observation_model == OBSERVATION_MODELS.SEEG_LOGPOWER.value
    if log_flag:
        preprocessing.append("log")
    preprocessing.append("decimate")
    pse_flag = True
    fit_flag = True
    if EMPIRICAL:
        main_fit_sim_hyplsa(stan_model_name=stan_model_name, observation_model=observation_model,
                            empirical_file=os.path.join(config.input.RAW_DATA_FOLDER, seizure),
                            sensors_lbls=sensors_lbls, times_on_off=times_on_off, sim_times_on_off=sim_times_on_off,
                            preprocessing_sequence=preprocessing, fitmethod=fitmethod,
                            pse_flag=pse_flag, fit_flag=fit_flag, config=config)
    else:
        main_fit_sim_hyplsa(stan_model_name=stan_model_name, observation_model=observation_model,
                            sensors_lbls=sensors_lbls, times_on_off=times_on_off, sim_times_on_off=sim_times_on_off,
                            preprocessing_sequence=preprocessing, fitmethod=fitmethod,
                            pse_flag=pse_flag, fit_flag=fit_flag, config=config)
