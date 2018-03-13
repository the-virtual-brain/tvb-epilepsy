# encoding=utf8

import os
import numpy as np
from scipy.io import loadmat, savemat
from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.constants.model_constants import K_DEF
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, ensure_list
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.service.model_inversion.sde_model_inversion_service import SDEModelInversionService
from tvb_epilepsy.service.model_inversion.stan.cmdstan_service import CmdStanService
from tvb_epilepsy.service.model_inversion.stan.pystan_service import PyStanService
from tvb_epilepsy.service.model_inversion.vep_stan_dict_builder import build_stan_model_dict
from tvb_epilepsy.service.model_inversion.vep_stan_dict_builder import build_stan_model_dict_to_interface_ins
from tvb_epilepsy.top.scripts.hypothesis_scripts import from_hypothesis_to_model_config_lsa
from tvb_epilepsy.top.scripts.simulation_scripts import from_model_configuration_to_simulation
from tvb_epilepsy.top.scripts.fitting_data_scripts import prepare_seeg_observable_from_mne_file


def main_fit_sim_hyplsa(stats_model_name="vep_sde", empirical_file=None, dynamical_model = "EpileptorDP2D",
                        times_on_off=[], time_units="msec", sensors_lbls=[], sensors_inds=[], fitmethod="optimizing",
                        stan_service="CmdStan", config=Config(), **kwargs):
    logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
    reader = H5Reader()
    writer = H5Writer()
    plotter = Plotter(config)
    # ------------------------------Stan model and service--------------------------------------
    # Compile or load model:
    # model_code_path = os.path.join(STATS_MODELS_PATH, stats_model_name + ".stan")
    model_code_path = os.path.join(config.generic.STATS_MODELS_PATH, stats_model_name + ".stan")
    if isequal_string(stan_service, "CmdStan"):
        stan_service = CmdStanService(model_name=stats_model_name, model=None, model_code=None,
                                      model_code_path=model_code_path,
                                      fitmethod=fitmethod, random_seed=12345, init="random", config=config)
    else:
        stan_service = PyStanService(model_name=stats_model_name, model=None, model_code=None,
                                     model_code_path=model_code_path,
                                     fitmethod=fitmethod, random_seed=12345, init="random", config=config)
    stan_service.set_or_compile_model()

    # -------------------------------Reading data-----------------------------------
    logger.info("Reading from: " + config.input.HEAD)
    head = reader.read_head(config.input.HEAD)
    # plotter.plot_head(head)

    # Formulate a VEP hypothesis manually
    hyp_builder = HypothesisBuilder(head.connectivity.number_of_regions, config).set_normalize(0.99)

    # Regions of Pathological Excitability hypothesis:
    x0_indices = [2, 25]
    x0_values = [0.01, 0.01]
    hyp_builder.set_x0_hypothesis(x0_indices, x0_values)

    # Regions of Model Epileptogenicity hypothesis:
    e_indices = list(range(head.connectivity.number_of_regions))
    e_indices.remove(2)
    e_indices.remove(25)
    e_values = np.zeros((head.connectivity.number_of_regions,)) + 0.01
    e_values[[1, 26]] = 0.99
    e_values = np.delete(e_values, [2, 25]).tolist()
    print(e_indices, e_values)
    hyp_builder.set_e_hypothesis(e_indices, e_values)
    # K_unscaled = 5.0 * K_DEF
    # Regions of Connectivity hypothesis:
    # w_indices = []  # [(0, 1), (0, 2)]
    # w_values = []  # [0.5, 2.0]
    # hypo_builder.set_w_indices(w_indices).set_w_values(w_values)

    hypothesis1 = hyp_builder.build_hypothesis()

    e_indices = [1, 26]  # [1, 2, 25, 26]
    hypothesis2 = hyp_builder.build_hypothesis_from_file("clinical_hypothesis_postseeg", e_indices)
    # Change something manually if necessary
    # hypothesis2.x0_values = [0.01, 0.01]
    K_unscaled = 3.0 * K_DEF

    hypos = (hypothesis1, hypothesis2)

    for hyp in hypos[1:]:

        # --------------------------Model configuration and LSA-----------------------------------
        model_config_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_ModelConfig.h5")
        hyp_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_LSA.h5")
        if os.path.isfile(hyp_file) and os.path.isfile(model_config_file):
            model_configuration = reader.read_model_configuration(model_config_file)
            lsa_hypothesis = reader.read_hypothesis(hyp_file)
        else:
            model_configuration, lsa_hypothesis, model_configuration_builder, lsa_service = \
                from_hypothesis_to_model_config_lsa(hyp, head, eigen_vectors_number=None, weighted_eigenvector_sum=True,
                                                    config=config, K=K_unscaled)
            writer.write_model_configuration(model_configuration, model_config_file)
            writer.write_hypothesis(lsa_hypothesis, hyp_file)
            plotter.plot_state_space(model_configuration, "6d", head.connectivity.region_labels,
                                     special_idx=hyp.get_regions_disease_indices(), zmode="lin",
                                     figure_name=hyp.name + "_StateSpace")
            plotter.plot_lsa(lsa_hypothesis, model_configuration, lsa_service.weighted_eigenvector_sum,
                             lsa_service.eigen_vectors_number, head.connectivity.region_labels, None)

        # -------------------------- Get model_data and observation signals: -------------------------------------------
        model_inversion_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_ModelInversionService.h5")
        stats_model_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_StatsModel.h5")
        model_data_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_ModelData.h5")
        if os.path.isfile(model_inversion_file) and os.path.isfile(stats_model_file) \
                and os.path.isfile(model_data_file):
            model_inversion = reader.read_model_inversions_service(model_inversion_file)
            statistical_model = reader.read_generic(stats_model_file)
            model_data = stan_service.load_model_data_from_file(model_data_path=model_data_file)
        else:
            model_inversion = SDEModelInversionService(model_configuration, lsa_hypothesis, head, dynamical_model,
                                                       x1eq_max=-1.0, sig=0.05, priors_mode="uninformative")
            # observation_expression="lfp"
            observation_model = "seeg_logpower"
            statistical_model = model_inversion.generate_statistical_model(x1eq_max=-1.0,
                                                                           observation_model=observation_model)
            statistical_model = model_inversion.update_active_regions(statistical_model, methods=["e_values", "LSA"],
                                                                      active_regions_th=0.2, reset=True)
            # plotter.plot_statistical_model(statistical_model, "Statistical Model")
            n_electrodes = 8
            sensors_per_electrode = 2
            if os.path.isfile(empirical_file):
                # ---------------------------------------Get empirical data-------------------------------------------
                target_data_type = "empirical"
                statistical_model.observation_model = "seeg_logpower"
                ts_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_ts_empirical.mat")
                try:
                    vois_ts_dict = loadmat(ts_file)
                    time = vois_ts_dict["time"].flatten()
                    sensors_inds = np.array(vois_ts_dict["sensors_inds"]).flatten().tolist()
                    sensors_lbls = np.array(vois_ts_dict["sensors_lbls"]).flatten().tolist()
                    vois_ts_dict.update({"time": time, "sensors_inds": sensors_inds, "sensors_lbls": sensors_lbls})
                    savemat(ts_file, vois_ts_dict)
                except:
                    if len(sensors_lbls) == 0:
                        sensors_lbls = head.get_sensors_id().labels
                    signals, time, sensors_inds = \
                        prepare_seeg_observable_from_mne_file(empirical_file, dynamical_model, times_on_off,
                                                              sensors_lbls, sensors_inds, time_units=time_units,
                                                              win_len_ratio=10, plotter=plotter)[:3]
                    inds = np.argsort(sensors_inds)
                    sensors_inds = np.array(sensors_inds)[inds].flatten().tolist()
                    model_inversion.sensors_labels = np.array(sensors_lbls).flatten().tolist()
                    all_signals = np.zeros((signals.shape[0], len(model_inversion.sensors_labels)))
                    all_signals[:, sensors_inds] = signals[:, inds]
                    signals = all_signals
                    del all_signals
                    vois_ts_dict = {"time": time.flatten(), "signals": signals,
                                    "sensors_inds": sensors_inds, "sensors_lbls": sensors_lbls}
                    savemat(ts_file, vois_ts_dict)
                manual_selection = sensors_inds
            else:
                # -------------------------- Get simulated data (simulate if necessary) -------------------------------
                target_data_type = "seeg_logpower"
                statistical_model.observation_model = observation_model
                ts_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_ts.mat")
                vois_ts_dict = \
                    from_model_configuration_to_simulation(model_configuration, head, lsa_hypothesis,
                                                           sim_type="realistic", dynamical_model=dynamical_model,
                                                           ts_file=ts_file, plot_flag=True, config=config)
                # if len(sensors_inds) > 1:  # get_bipolar_channels(sensors_inds, sensors_lbls)
                #     sensors_inds, sensors_lbls = head.get_sensors_id().get_bipolar_sensors(sensors_inds=sensors_inds)
                if statistical_model.observation_model.find("seeg") >= 0:
                    manual_selection = sensors_inds
                else:
                    if stats_model_name.find("vep-fe-rev") >= 0:
                        manual_selection = statistical_model.active_regions
                    else:
                        manual_selection = []
                # -------------------------- Select and set observation signals -----------------------------------
            signals, time, statistical_model, vois_ts_dict = \
                model_inversion.set_target_data_and_time(target_data_type, vois_ts_dict, statistical_model,
                                                         dynamical_model, times_on_off=times_on_off,
                                                         manual_selection=manual_selection,
                                                         auto_selection="correlation-power",  # auto_selection=False,
                                                         n_electrodes=n_electrodes,
                                                         sensors_per_electrode=sensors_per_electrode,
                                                         group_electrodes=True, normalization="baseline-amplitude",
                                                         plotter=plotter)
            # if len(model_inversion.signals_inds) < head.get_sensors_id().number_of_sensors:
            #     statistical_model = \
            #             model_inversion.update_active_regions_seeg(statistical_model)
            if model_inversion.data_type == "lfp":
                labels = model_inversion.region_labels
                special_idx = []
            else:
                labels = model_inversion.sensors_labels
            if vois_ts_dict.get("signals", None) is not None:
                vois_ts_dict["signals"] -= vois_ts_dict["signals"].min()
                vois_ts_dict["signals"] /= vois_ts_dict["signals"].max()
                if statistical_model.observation_model == "seeg_logpower":
                    special_idx = model_inversion.signals_inds
                else:
                    special_idx = []
                plotter.plot_raster({'Target Signals': vois_ts_dict["signals"]}, vois_ts_dict["time"].flatten(),
                                    time_units="ms", title=hyp.name + ' Target Signals raster',
                                    special_idx=special_idx, offset=0.1, labels=labels)
            plotter.plot_timeseries({'Target Signals': signals}, time, time_units="ms",
                                    title=hyp.name + ' Target Signals',
                                    labels=labels[model_inversion.signals_inds])
            writer.write_model_inversion_service(model_inversion, os.path.join(config.out.FOLDER_RES,
                                                                               hyp.name + "_ModelInversionService.h5"))
            writer.write_generic(statistical_model, config.out.FOLDER_RES, hyp.name + "_StatsModel.h5")
            # try:
            #     model_data = stan_service.load_model_data_from_file()
            # except:
            model_data = build_stan_model_dict(statistical_model, signals, model_inversion)
            writer.write_dictionary(model_data, os.path.join(config.out.FOLDER_RES, hyp.name + "_ModelData.h5"))

        simulation_values = {"x0": model_configuration.x0, "x1eq": model_configuration.x1EQ,
                             "x1init": model_configuration.x1EQ, "zinit": model_configuration.zEQ}
        # Stupid code to interface with INS stan model
        if stats_model_name.find("vep-fe-rev") >= 0:
            model_data, x0_star_mu, x_init_mu, z_init_mu = \
                build_stan_model_dict_to_interface_ins(model_data, statistical_model, model_inversion,
                                                       informative_priors=False)
            x1_str = "x"
            input_signals_str = "seeg_log_power"
            signals_str = "mu_seeg_log_power"
            dX1t_str = "x_eta"
            dZt_str = "z_eta"
            sig_str = "sigma"
            k_str = "k"
            pair_plot_params = ["time_scale", "k", "sigma", "epsilon", "amplitude", "offset"]
            region_violin_params = ["x0", "x_init", "z_init"]
            if empirical_file:
                priors = {"x0": x0_star_mu, "x_init": x_init_mu, "z_init": z_init_mu}
            else:
                priors = dict(simulation_values)
                priors.update({"x0": simulation_values["x0"][statistical_model.active_regions]})
                priors.update({"x_init": simulation_values["x1init"][statistical_model.active_regions]})
                priors.update({"z_init": simulation_values["zinit"][statistical_model.active_regions]})
            connectivity_plot = False
            estMC = lambda: model_configuration.model_connectivity
            region_mode = "active"
        else:
            x1_str = "x1"
            input_signals_str = "signals"
            signals_str = "fit_signals"
            dX1t_str = "dX1t"  # "x1_dWt"
            dZt_str = "dZt"  # "z_dWt"
            sig_str = "sig"
            k_str = "K"
            pair_plot_params = ["tau1", "tau0", "K", "sig_init", "sig", "eps", "scale_signal", "offset_signal"]
            region_violin_params = ["x0", "x1eq", "x1init", "zinit"]
            connectivity_plot = False
            estMC = lambda est: est["MC"]
            region_mode = "all"
            if empirical_file:
                priors = {"x0": model_inversion.x0[statistical_model.active_regions],
                          "x1eq": model_data["x1eq_max"]
                                  - statistical_model.parameters["x1eq_star"].mean[statistical_model.active_regions],
                          "x_init": statistical_model.parameters["x1init"].mean[statistical_model.active_regions],
                          "z_init": statistical_model.parameters["zinit"].mean[statistical_model.active_regions]}
            else:
                priors = simulation_values

        # -------------------------- Fit and get estimates: ------------------------------------------------------------
        fit=True
        num_warmup = 200
        if fit:
            ests, samples, summary = stan_service.fit(debug=0, simulate=0, model_data=model_data, merge_outputs=False,
                                                      chains=4, refresh=1, num_warmup=num_warmup, num_samples=300,
                                                      max_depth=10, delta=0.8, save_warmup=1, plot_warmup=1, **kwargs)
            writer.write_generic(ests, config.out.FOLDER_RES, hyp.name + "_fit_est.h5")
            writer.write_generic(samples, config.out.FOLDER_RES, hyp.name + "_fit_samples.h5")
            if summary is not None:
                writer.write_generic(summary, config.out.FOLDER_RES, hyp.name + "_fit_summary.h5")
                if isinstance(summary, dict):
                    R_hat = summary.get("R_hat", None)
                    if R_hat is not None:
                        R_hat = {"R_hat": R_hat}
        else:
            ests, samples, summary = stan_service.read_output()
            if isinstance(summary, dict):
                R_hat = summary.get("R_hat", None)
                if R_hat is not None:
                    R_hat = {"R_hat": R_hat}
            if fitmethod.find("sampl") >= 0:
                Plotter(config).plot_HMC(samples, skip_samples=num_warmup)
        ests = ensure_list(ests)
        plotter.plot_fit_results(model_inversion, ests, samples, statistical_model, model_data[input_signals_str],
                                 R_hat, model_data["time"], priors, region_mode,
                                 seizure_indices=lsa_hypothesis.get_regions_disease_indices(), x1_str=x1_str,
                                 k_str=k_str, signals_str=signals_str, sig_str=sig_str, dX1t_str=dX1t_str,
                                 dZt_str=dZt_str, trajectories_plot=True, connectivity_plot=connectivity_plot,
                                 pair_plot_params=pair_plot_params, region_violin_params=region_violin_params)
        # -------------------------- Reconfigure model after fitting:---------------------------------------------------
        for id_est, est in enumerate(ensure_list(ests)):
            fit_model_configuration_builder = \
                ModelConfigurationBuilder(hyp.number_of_regions, K=est[k_str] * hyp.number_of_regions)
            x0_values_fit = model_configuration.x0_values
            x0_values_fit[statistical_model.active_regions] = \
                fit_model_configuration_builder._compute_x0_values_from_x0_model(est['x0'])
            hyp_fit = HypothesisBuilder().set_nr_of_regions(head.connectivity.number_of_regions).\
                                          set_name('fit' + str(id_est) + "_" + hyp.name).\
                                          set_x0_hypothesis(list(statistical_model.active_regions),
                                                            x0_values_fit[statistical_model.active_regions]).\
                                          build_hypothesis()
            writer.write_hypothesis(hyp_fit, os.path.join(config.out.FOLDER_RES, hyp_fit.name + ".h5"))

            model_configuration_fit = \
                fit_model_configuration_builder.build_model_from_hypothesis(hyp_fit,  # est["MC"]
                                                                            model_configuration.model_connectivity)

            writer.write_model_configuration(model_configuration_fit,
                                             os.path.join(config.out.FOLDER_RES, hyp_fit.name + "_ModelConfig.h5"))

            # Plot nullclines and equilibria of model configuration
            plotter.plot_state_space(model_configuration_fit,
                                     region_labels=model_inversion.region_labels,
                                     special_idx=statistical_model.active_regions,
                                     model="6d", zmode="lin",
                                     figure_name=hyp_fit.name + "_Nullclines and equilibria")
        logger.info("Done!")


if __name__ == "__main__":

    user_home = os.path.expanduser("~")
    head_folder = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "CC", "TVB3", "Head")
    SEEG_data = os.path.join(os.path.expanduser("~"), 'Dropbox', 'Work', 'VBtech', 'VEP', "data/CC", "TVB3",
                             "raw/seeg/ts_seizure")

    if user_home == "/home/denis":
        output = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "INScluster")
        config = Config(head_folder=head_folder, raw_data_folder=SEEG_data,
                        output_base=output, separate_by_run=True)
        config.generic.C_COMPILER = "g++"
        config.generic.CMDSTAN_PATH = "/soft/stan/cmdstan-2.17.0"

    elif user_home == "/Users/lia.domide":
        config = Config(head_folder="/WORK/episense/tvb-epilepsy/data/TVB3/Head",
                        raw_data_folder="/WORK/episense/tvb-epilepsy/data/TVB3/ts_seizure")
        config.generic.CMDSTAN_PATH = "/WORK/episense/cmdstan-2.17.1"

    else:
        output = os.path.join(user_home, 'Dropbox', 'Work', 'VBtech', 'VEP', "results", "fit")
        config = Config(head_folder=head_folder, raw_data_folder=SEEG_data,
                        output_base=output, separate_by_run=False)

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
    seizure = 'SZ1_0001.edf'
    times_on_off = [15.0, 35.0] * 1000
    # sensors_filename = "SensorsSEEG_116.h5"
    # # TVB4 preselection:
    # sensors_lbls = [u"D5", u"D6", u"D7",  u"D8", u"D9", u"D10", u"Z9", u"Z10", u"Z11", u"Z12", u"Z13", u"Z14",
    #                 u"S1", u"S2", u"S3", u"D'3", u"D'4", u"D'10", u"D'11", u"D'12", u"D'13", u"D'14"]
    # sensors_inds = [4, 5, 6, 7, 8, 9, 86, 87, 88, 89, 90, 91, 94, 95, 96, 112, 113, 119, 120, 121, 122, 123]
    # # TVB4:
    # seizure = 'SZ3_0001.edf'
    # sensors_filename = "SensorsSEEG_210.h5"
    # times_on_off = [20.0, 100.0]
    EMPIRICAL = True
    # stats_model_name = "vep_sde"
    stats_model_name = "vep-fe-rev-09dp"
    fitmethod = "sample"
    if EMPIRICAL:
        main_fit_sim_hyplsa(stats_model_name=stats_model_name,
                            empirical_file=os.path.join(config.input.RAW_DATA_FOLDER, seizure),
                            times_on_off=times_on_off, time_units="sec", sensors_inds=sensors_inds,
                            fitmethod=fitmethod, stan_service="CmdStan", config=config)
    else:
        main_fit_sim_hyplsa(stats_model_name=stats_model_name, times_on_off=[1000.0, 19000.0],
                            sensors_inds=sensors_inds,
                            fitmethod=fitmethod, stan_service="CmdStan", config=config)
