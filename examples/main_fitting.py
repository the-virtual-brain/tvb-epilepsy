# encoding=utf8

import os
import numpy as np
from scipy.io import loadmat, savemat
from tvb_epilepsy.base.constants.configurations import FOLDER_RES, DATA_CUSTOM, FOLDER_FIGURES, FOLDER_VEP_ONLINE
from tvb_epilepsy.base.constants.module_constants import TVB, CUSTOM
from tvb_epilepsy.base.h5_model import convert_to_h5_model, read_h5_model
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, ensure_list
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.plot_utils import plot_raster, plot_timeseries
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.io.h5.writer_custom import CustomH5Writer
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.service.model_inversion.sde_model_inversion_service import SDEModelInversionService
from tvb_epilepsy.service.model_inversion.stan.cmdstan_service import CmdStanService
from tvb_epilepsy.service.model_inversion.stan.pystan_service import PyStanService
from tvb_epilepsy.scripts.hypothesis_scripts import from_head_to_hypotheses, from_hypothesis_to_model_config_lsa
from tvb_epilepsy.scripts.simulation_scripts import from_model_configuration_to_simulation
from tvb_epilepsy.scripts.seeg_data_scripts import prepare_seeg_observable

logger = initialize_logger(__name__)

FOLDER_VEP_HOME = os.path.join(FOLDER_VEP_ONLINE, "tests")


def main_fit_sim_hyplsa(ep_name="ep_l_frontal_complex", data_folder=os.path.join(DATA_CUSTOM, 'Head'),
                        sensors_filename="SensorsSEEG_116.h5", stats_model_name="vep_sde",
                        model_code_dir="/Users/dionperd/VEPtools/git/tvb-epilepsy/tvb_epilepsy/stan", EMPIRICAL="",
                        times_on_off=[], sensors_lbls=[], sensors_inds=[], fitmethod="optimizing",
                        stan_service="CmdStan", results_dir=FOLDER_RES, figure_dir=FOLDER_FIGURES, **kwargs):
    # ------------------------------Stan model and service--------------------------------------
    # Compile or load model:
    # model_code_path = os.path.join(STATS_MODELS_PATH, stats_model_name + ".stan")
    model_code_path = os.path.join(model_code_dir, stats_model_name + ".stan")
    if isequal_string(stan_service, "CmdStan"):
        stan_service = CmdStanService(model_name=stats_model_name, model=None, model_code=None,
                                      model_dir=FOLDER_VEP_HOME, model_code_path=model_code_path,
                                      fitmethod=fitmethod, random_seed=12345, init="random", logger=logger)
    else:
        stan_service = PyStanService(model_name=stats_model_name, model=None, model_code=None,
                                     model_dir=FOLDER_VEP_HOME, model_code_path=model_code_path,
                                     fitmethod=fitmethod, random_seed=12345, init="random", logger=logger)
    stan_service.set_or_compile_model()

    # -------------------------------Reading model_data and hypotheses--------------------------------------------------
    head, hypos = from_head_to_hypotheses(ep_name, data_mode=CUSTOM, data_folder=data_folder,
                                          plot_head=False, figure_dir=figure_dir, sensors_filename=sensors_filename,
                                          logger=logger)

    for hyp in hypos:
        # --------------------------Model configuration and LSA-----------------------------------
        model_configuration, lsa_hypothesis, model_configuration_service, lsa_service = \
            from_hypothesis_to_model_config_lsa(hyp, head, eigen_vectors_number=None, weighted_eigenvector_sum=True,
                                                plot_flag=True, figure_dir=figure_dir, logger=logger, K=1.0)

        dynamical_model = "EpileptorDP2D"

        # -------------------------- Get model_data and observation signals: -------------------------------------------
        model_inversion = SDEModelInversionService(model_configuration, lsa_hypothesis, head, dynamical_model,
                                                   sde_mode="x1z", logger=logger)
        statistical_model = model_inversion.generate_statistical_model(observation_expression="lfp")
        statistical_model = model_inversion.update_active_regions(statistical_model, methods=["e_values", "LSA"],
                                                                  active_regions_th=0.1, reset=True)
        decimate = 4
        cut_signals_tails = (6, 6)
        if os.path.isfile(EMPIRICAL):
            # ---------------------------------------Get empirical data----------------------------------------------
            target_data_type = "empirical"
            ts_file = os.path.join(FOLDER_VEP_HOME, lsa_hypothesis.name + "_ts_empirical.mat")
            try:
                # vois_ts_dict = read_h5_model(ts_file).convert_from_h5_model()
                vois_ts_dict = loadmat(ts_file)
                time = vois_ts_dict["time"].flatten()
                sensors_inds = np.array(vois_ts_dict["sensors_inds"]).flatten().tolist()
                sensors_lbls = np.array(vois_ts_dict["sensors_lbls"]).flatten().tolist()
                vois_ts_dict.update({"time": time, "sensors_inds": sensors_inds, "sensors_lbls": sensors_lbls})
                # convert_to_h5_model(vois_ts_dict).write_to_h5(FOLDER_VEP_HOME, lsa_hypothesis.name + "_ts_empirical.h5")
                savemat(ts_file, vois_ts_dict)
            except:
                signals, time, fs = prepare_seeg_observable(EMPIRICAL, times_on_off, sensors_lbls, plot_flag=True,
                                                            log_flag=True)
                if len(sensors_inds) > 1:  # get_bipolar_channels(sensors_inds, sensors_lbls)
                    sensors_inds, sensors_lbls = head.get_sensors_id().get_bipolar_sensors(sensors_inds=sensors_inds)
                inds = np.argsort(sensors_inds)
                sensors_inds = np.array(sensors_inds)[inds].flatten().tolist()
                sensors_lbls = np.array(sensors_lbls)[inds].flatten().tolist()
                all_signals = np.zeros((signals.shape[0], len(model_inversion.sensors_labels)))
                all_signals[:, sensors_inds] = signals[:, inds]
                signals = all_signals
                del all_signals
                vois_ts_dict = {"time": time.flatten(), "signals": signals,
                                "sensors_inds": sensors_inds, "sensors_lbls": sensors_lbls}
                # convert_to_h5_model(vois_ts_dict).write_to_h5(FOLDER_VEP_HOME, lsa_hypothesis.name + "_ts_empirical.h5")
                savemat(ts_file, vois_ts_dict)
            model_inversion.sensors_labels[vois_ts_dict["sensors_inds"]] = sensors_lbls
            manual_selection = sensors_inds
            n_electrodes = 4
            sensors_per_electrode = 2
        else:
            # -------------------------- Get simulated data (simulate if necessary) -----------------------------------
            target_data_type = "simulated"
            ts_file = os.path.join(FOLDER_VEP_HOME, lsa_hypothesis.name + "_ts.h5")
            vois_ts_dict = \
                from_model_configuration_to_simulation(model_configuration, head, lsa_hypothesis, simulation_mode=TVB,
                                                       sim_type="fitting", dynamical_model=dynamical_model,
                                                       ts_file=ts_file, plot_flag=True,
                                                       save_flag=True, results_dir=results_dir,
                                                       figure_dir=figure_dir, logger=logger, tau1=0.5, tau0=30.0,
                                                       noise_intensity=10 ** -3)
            manual_selection = []
            n_electrodes = 8
            sensors_per_electrode = 1
            sensors_lbls = model_inversion.sensors_labels
        # -------------------------- Select and set observation signals -----------------------------------
        signals, time, statistical_model, vois_ts_dict = \
            model_inversion.set_target_data_and_time(target_data_type, vois_ts_dict, statistical_model,
                                                     select_signals=True, manual_selection=manual_selection,
                                                     # auto_selection=False,
                                                     n_electrodes=n_electrodes, auto_selection="correlation-power",
                                                     sensors_per_electrode=sensors_per_electrode, group_electrodes=True,
                                                     decimate=decimate, cut_signals_tails=cut_signals_tails)
        # if len(model_inversion.signals_inds) < head.get_sensors_id().number_of_sensors:
        #     statistical_model = \
        #             model_inversion.update_active_regions_seeg(statistical_model)
        if model_inversion.data_type == "lfp":
            labels = model_inversion.region_labels
        else:
            labels = model_inversion.sensors_labels
        if vois_ts_dict.get("signals", None) is not None:
            vois_ts_dict["signals"] -= vois_ts_dict["signals"].min()
            vois_ts_dict["signals"] /= vois_ts_dict["signals"].max()
            plot_raster(vois_ts_dict["time"].flatten(), {'Target Signals': vois_ts_dict["signals"]},
                        time_units="ms", title=hyp.name + ' Target Signals raster',
                        special_idx=model_inversion.signals_inds, offset=0.1, labels=labels,
                        save_flag=True, show_flag=False, figure_dir=figure_dir)
        plot_timeseries(time, {'Target Signals': signals},
                        time_units="ms", title=hyp.name + 'Target Signals ',
                        labels=labels[model_inversion.signals_inds],
                        save_flag=True, show_flag=False, figure_dir=figure_dir)
        writer = CustomH5Writer()
        writer.write_model_inversion_service(model_inversion, os.path.join(FOLDER_RES,
                                                                           lsa_hypothesis.name + "_ModelInversionService.h5"))
        statistical_model.write_to_h5(results_dir, lsa_hypothesis.name + "_StatsModel.h5")
        # try:
        #     model_data = stan_service.load_model_data_from_file()
        # except:
        model_data = model_inversion.generate_model_data(statistical_model, signals)
        convert_to_h5_model(model_data).write_to_h5(results_dir, "dpModelData.h5")

        if stats_model_name == "vep-fe-rev-05":
            def convert_to_vep_stan(model_data, statistical_model):
                active_regions = model_data["active_regions"]
                SC = statistical_model.parameters["MC"].mode[active_regions][:, active_regions]
                vep_data = {"nn": model_data["n_active_regions"],
                            "nt": model_data["n_times"],
                            "ns": model_data["n_signals"],
                            "dt": 0.75,  # model_data["dt"],
                            "I1": model_data["Iext1"],
                            "x0_lo": -3.0,
                            "x0_hi": -1.0,
                            "tau0": 3.0,  # statistical_model.parameters["tau0"].mean,
                            "K_lo": statistical_model.parameters["K"].low,
                            "K_u": statistical_model.parameters["K"].mode,
                            "K_v": statistical_model.parameters["K"].var,
                            "SC": SC,
                            "SC_var": 5.0,  # 1/36 = 0.02777777,
                            "Ic": np.sum(SC, axis=1),
                            "sig_hi": 0.025,  # model_data["sig_hi"],
                            "gain": model_data["mixing"],
                            "seeg_log_power": 9.0 * model_data["signals"] - 4.0,  # scale from (0, 1) to (-4, 5)
                            }
                return vep_data

            model_data = convert_to_vep_stan(model_data, statistical_model)
        stan_service.write_model_data_to_file(model_data)

        # -------------------------- Fit and get estimates: ------------------------------------------------------------
        est, fit = stan_service.fit(model_data=model_data, debug=1, simulate=0,
                                    merge_outputs=False, chains=1, refresh=1, **kwargs)
        convert_to_h5_model(est).write_to_h5(results_dir, lsa_hypothesis.name + "_fit_est.h5")
        est = ensure_list(est)
        for id_est, this_est in enumerate(est):
            model_inversion.plot_fit_results(this_est, statistical_model, signals, time=None,
                                             seizure_indices=lsa_hypothesis.get_regions_disease(),
                                             trajectories_plot=True, id_est=str(id_est))
            # -------------------------- Reconfigure model after fitting:---------------------------------------------------
            fit_model_configuration_service = \
                ModelConfigurationService(hyp.number_of_regions, K=this_est['K'] * hyp.number_of_regions)
            x0_values_fit = \
                fit_model_configuration_service._compute_x0_values_from_x0_model(this_est['x0'])
            hyp_fit = \
                DiseaseHypothesis(head.connectivity.number_of_regions,
                                  excitability_hypothesis={tuple(range(model_configuration.n_regions)): x0_values_fit},
                                  epileptogenicity_hypothesis={}, connectivity_hypothesis={},
                                  name='fit' + str(id_est) + "_" + hyp.name)
            model_configuration_fit = fit_model_configuration_service.configure_model_from_hypothesis(hyp_fit,
                                                                                                      this_est["MC"])
            writer.write_model_configuration(model_configuration_fit,
                                             os.path.join(results_dir, hyp_fit.name + "_ModelConfig.h5"))

            # Plot nullclines and equilibria of model configuration
            model_configuration_service.plot_state_space(model_configuration_fit,
                                                         model_configuration_service.region_labels,
                                                         special_idx=statistical_model.active_regions,
                                                         model="6d", zmode="lin",
                                                         figure_name=hyp_fit.name + "_Nullclines and equilibria")
        print("Done!")


if __name__ == "__main__":
    SUBJECT = "TVB3"
    VEP_HOME = os.path.join("/Users/dionperd/Dropbox/Work/VBtech/VEP/results/CC")
    VEP_FOLDER = os.path.join(VEP_HOME, SUBJECT)
    DATA_CUSTOM = "/Users/dionperd/Dropbox/Work/VBtech/VEP/results/CC/" + SUBJECT
    SEEG_data = os.path.join("/Users/dionperd/Dropbox/Work/VBtech/VEP/data/CC", SUBJECT, "raw/seeg/ts_seizure")
    # TVB3 preselection:
    # sensors_lbls = [u"G'1", u"G'2", u"G'3", u"G'8", u"G'9", u"G'10", u"G'11", u"G'12", u"M'6", u"M'7", u"M'8", u"L'4",
    #                 u"L'5",  u"L'6", u"L'7", u"L'8", u"L'9"]
    # sensors_inds = [28, 29, 30, 35, 36, 37, 38, 39, 63, 64, 65, 47, 48, 49, 50, 51, 52]
    # TVB3 selection:
    sensors_lbls = [u"G'1", u"G'2", u"G'11", u"G'12", u"M'7", u"M'8", u"L'5", u"L'6"]
    sensors_inds = [28, 29, 38, 39, 64, 65, 48, 49]
    seizure = 'SZ1_0001.edf'
    times_on_off = [15.0, 35.0]
    ep_name = "clinical_hypothesis_postseeg"
    sensors_filename = "SensorsSEEG_116.h5"
    # # TVB4 preselection:
    # sensors_lbls = [u"D5", u"D6", u"D7",  u"D8", u"D9", u"D10", u"Z9", u"Z10", u"Z11", u"Z12", u"Z13", u"Z14",
    #                 u"S1", u"S2", u"S3", u"D'3", u"D'4", u"D'10", u"D'11", u"D'12", u"D'13", u"D'14"]
    # sensors_inds = [4, 5, 6, 7, 8, 9, 86, 87, 88, 89, 90, 91, 94, 95, 96, 112, 113, 119, 120, 121, 122, 123]
    # # TVB4:
    # seizure = 'SZ3_0001.edf'
    # sensors_filename = "SensorsSEEG_210.h5"
    # times_on_off = [20.0, 100.0]
    # ep_name = "clinical_hypothesis_preseeg_right"
    EMPIRICAL = True
    stats_model_name = "vep_sde"
    # stats_model_name = "vep-fe-rev-05"
    fitmethod = "sample"
    model_code_dir = "/Users/dionperd/VEPtools/git/tvb-epilepsy/tvb_epilepsy/stan"
    if EMPIRICAL:
        main_fit_sim_hyplsa(ep_name=ep_name, data_folder=os.path.join(DATA_CUSTOM, 'Head'),
                            sensors_filename=sensors_filename, stats_model_name=stats_model_name,
                            model_code_dir=model_code_dir, EMPIRICAL=os.path.join(SEEG_data, seizure),
                            times_on_off=[15.0, 35.0], sensors_lbls=sensors_lbls, sensors_inds=sensors_inds,
                            fitmethod=fitmethod, stan_service="CmdStan", results_dir=FOLDER_RES,
                            figure_dir=FOLDER_FIGURES, save_warmup=1, num_warmup=200, num_samples=200, delta=0.8,
                            max_depth=7)  # , stan_service="PyStan"
    else:
        main_fit_sim_hyplsa(ep_name=ep_name, data_folder=os.path.join(DATA_CUSTOM, 'Head'),
                            sensors_filename=sensors_filename, stats_model_name=stats_model_name,
                            model_code_dir=model_code_dir, fitmethod=fitmethod, stan_service="CmdStan",
                            results_dir=FOLDER_RES, figure_dir=FOLDER_FIGURES)  # , stan_service="PyStan"
