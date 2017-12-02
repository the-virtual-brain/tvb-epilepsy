# encoding=utf8

import os

import numpy as np

from tvb_epilepsy.base.constants.configurations import USER_HOME, FOLDER_RES, DATA_CUSTOM, STATS_MODELS_PATH, \
                                                                                        FOLDER_VEP_HOME, FOLDER_FIGURES
from tvb_epilepsy.base.constants.module_constants import TVB, CUSTOM
from tvb_epilepsy.base.h5_model import convert_to_h5_model, read_h5_model
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, ensure_list
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.plot_utils import plot_raster, plot_timeseries
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.service.model_inversion.sde_model_inversion_service import SDEModelInversionService
from tvb_epilepsy.service.model_inversion.stan.cmdstan_service import CmdStanService
from tvb_epilepsy.service.model_inversion.stan.pystan_service import PyStanService
from tvb_epilepsy.scripts.hypothesis_scripts import from_head_to_hypotheses, from_hypothesis_to_model_config_lsa
from tvb_epilepsy.scripts.simulation_scripts import from_model_configuration_to_simulation
from tvb_epilepsy.scripts.seeg_data_scripts import prepare_seeg_observable, get_bipolar_channels


logger = initialize_logger(__name__)


def main_fit_sim_hyplsa(ep_name="ep_l_frontal_complex", data_folder=os.path.join(DATA_CUSTOM, 'Head'),
                        stats_model_name="vep_sde", EMPIRICAL="", times_on_off=[], channel_lbls=[],
                        channel_inds=[], fitmethod="optimizing", stan_service="CmdStan", results_dir=FOLDER_RES,
                        figure_dir=FOLDER_FIGURES, **kwargs):

    # ------------------------------Stan model and service--------------------------------------
    # Compile or load model:
    if isequal_string(stan_service, "CmdStan"):
        stan_service = CmdStanService(model_name=stats_model_name, model=None, model_code=None,
                                      model_dir=FOLDER_VEP_HOME,
                                      model_code_path=os.path.join(STATS_MODELS_PATH, stats_model_name + ".stan"),
                                      fitmethod=fitmethod, random_seed=12345, init="random", logger=logger)
    else:
        stan_service = PyStanService(model_name=stats_model_name, model=None, model_code=None,
                                     model_dir=FOLDER_VEP_HOME,
                                     model_code_path=os.path.join(STATS_MODELS_PATH, stats_model_name + ".stan"),
                                     fitmethod=fitmethod, random_seed=12345, init="random", logger=logger)
    stan_service.set_or_compile_model()

    # -------------------------------Reading model_data and hypotheses--------------------------------------------------
    head, hypos = from_head_to_hypotheses(ep_name, data_mode=CUSTOM, data_folder=data_folder,
                                          plot_head=False, figure_dir=figure_dir, logger=logger)

    for hyp in hypos:
        # --------------------------Model configuration and LSA-----------------------------------
        model_configuration, lsa_hypothesis, model_configuration_service, lsa_service = \
            from_hypothesis_to_model_config_lsa(hyp, head, eigen_vectors_number=None, weighted_eigenvector_sum=True,
                                                plot_flag=True, figure_dir=figure_dir, logger=logger, K=10.0)

        dynamical_model = "EpileptorDP2D"

        # -------------------------- Get model_data and observation signals: -------------------------------------------
        model_inversion = SDEModelInversionService(model_configuration, lsa_hypothesis, head,  dynamical_model,
                                                   sde_mode="x1z",  logger=logger)
        statistical_model = model_inversion.generate_statistical_model(observation_expression="lfp")
        statistical_model = model_inversion.update_active_regions(statistical_model, methods=["e_values", "LSA"],
                                                                                     active_regions_th=0.1, reset=True)
        decimate = 4
        cut_signals_tails = (6, 6)
        if os.path.isfile(EMPIRICAL):
            # ---------------------------------------Get empirical data----------------------------------------------
            target_data_type = "empirical"
            ts_file = os.path.join(FOLDER_VEP_HOME, lsa_hypothesis.name + "_ts_empirical.h5")
            try:
                vois_ts_dict = read_h5_model(ts_file).convert_from_h5_model()
                channel_inds = vois_ts_dict["channel_inds"]
                channel_lbls = vois_ts_dict["channel_lbls"]
            except:
                signals, time, fs = prepare_seeg_observable(EMPIRICAL, times_on_off, channel_lbls, log_flag=True)
                if len(channel_inds) > 1:
                    channel_inds, channel_lbls = get_bipolar_channels(channel_inds, channel_lbls)
                inds = np.argsort(channel_inds)
                channel_inds = np.array(channel_inds)[inds].tolist()
                channel_lbls = np.array(channel_lbls)[inds].tolist()
                all_signals = np.zeros((signals.shape[0], len(model_inversion.sensors_labels)))
                all_signals[:, channel_inds] = signals[:,inds]
                signals = all_signals
                del all_signals
                vois_ts_dict = {"time": time, "signals": signals,
                                "channel_inds": channel_inds, "channel_lbls": channel_lbls}
                convert_to_h5_model(vois_ts_dict).write_to_h5(FOLDER_VEP_HOME, lsa_hypothesis.name + "_ts_empirical.h5")
            model_inversion.sensors_labels[channel_inds] = channel_lbls
            manual_selection = channel_inds
            n_electrodes = 3
            contacts_per_electrode = 2
        else:
            # -------------------------- Get simulated data (simulate if necessary) -----------------------------------
            target_data_type = "simulated"
            ts_file = os.path.join(FOLDER_VEP_HOME, lsa_hypothesis.name + "_ts.h5")
            vois_ts_dict = \
                from_model_configuration_to_simulation(model_configuration, head, lsa_hypothesis,
                                                       dynamical_model=dynamical_model,
                                                       simulation_mode=TVB, ts_file=ts_file, plot_flag=True,
                                                       save_flag=True, results_dir=results_dir,
                                                       figure_dir=figure_dir, logger=logger, tau1=0.5, tau0=30.0,
                                                       noise_intensity=10**-2.8)
            manual_selection = []
            n_electrodes = 8
            contacts_per_electrode = 1
            channel_lbls = model_inversion.sensors_labels
        # -------------------------- Select and set observation signals -----------------------------------
        signals, time, statistical_model, vois_ts_dict = \
            model_inversion.set_target_data_and_time(target_data_type, vois_ts_dict, statistical_model,
                                                     select_signals=True, manual_selection=manual_selection,
                                                     n_electrodes=n_electrodes, auto_selection="rois-correlation-power",
                                                     contacts_per_electrode=contacts_per_electrode,
                                                     group_electrodes=False,
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
            plot_raster(vois_ts_dict['time'], {'Target Signals': vois_ts_dict["signals"]},
                        time_units="ms", title= hyp.name + ' Target Signals raster',
                        special_idx=model_inversion.signals_inds, offset=0.1, labels=labels,
                        save_flag=True, show_flag=False, figure_dir=figure_dir)
        plot_timeseries(time, {'Target Signals': signals},
                        time_units="ms", title= hyp.name + 'Target Signals ', 
                        labels=labels[model_inversion.signals_inds],
                        save_flag=True, show_flag=False, figure_dir=figure_dir)
        model_inversion.write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_ModelInversionService.h5")
        statistical_model.write_to_h5(results_dir, lsa_hypothesis.name + "_StatsModel.h5")
        # try:
        #     model_data = stan_service.load_model_data_from_file()
        # except:
        model_data = model_inversion.generate_model_data(statistical_model, signals)
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
                ModelConfigurationService(hyp.number_of_regions, K=this_est['K']*hyp.number_of_regions)
            x0_values_fit = \
                fit_model_configuration_service._compute_x0_values_from_x0_model(this_est['x0'])
            hyp_fit = \
                DiseaseHypothesis(head.connectivity.number_of_regions,
                                  excitability_hypothesis={tuple(range(model_configuration.n_regions)): x0_values_fit},
                                  epileptogenicity_hypothesis={}, connectivity_hypothesis={},
                                  name='fit' + str(id_est) + "_" + hyp.name)
            model_configuration_fit = fit_model_configuration_service.configure_model_from_hypothesis(hyp_fit,
                                                                                                      this_est["MC"])
            model_configuration_fit.write_to_h5(results_dir, hyp_fit.name + "_ModelConfig.h5")
            # Plot nullclines and equilibria of model configuration
            model_configuration_service.plot_nullclines_eq(model_configuration_fit,
                                                           model_configuration_service.region_labels,
                                                           special_idx=statistical_model.active_regions,
                                                           model="6d", zmode="lin",
                                                           figure_name=hyp_fit.name + "_Nullclines and equilibria")
        print("Done!")


if __name__ == "__main__":
    VEP_HOME = os.path.join(USER_HOME, 'CBR/VEP/CC')
    VEP_FOLDER = os.path.join(VEP_HOME, 'TVB3')
    CT = os.path.join(VEP_FOLDER, 'CT')
    SEEG = os.path.join(VEP_FOLDER, 'SEEG')
    SEEG_data = os.path.join(VEP_FOLDER, 'SEEG_data')
    channel_lbls = [u"G'1", u"G'2", u"G'3", u"G'8", u"G'9", u"G'10", u"G'11", u"G'12", u"M'6", u"M'7", u"M'8", u"L'4",
                u"L'5",  u"L'6", u"L'7", u"L'8", u"L'9"]
    channel_inds = [67, 68, 69, 74, 75, 76, 77, 78, 21, 22, 23, 43, 44, 45, 46,  47, 48]
    # -------------------------------Reading model_data-----------------------------------
    # data_folder = os.path.join(DATA_CUSTOM, 'Head')
    #
    # reader = Reader()
    #
    # logger.info("Reading from: " + data_folder)
    # head = reader.read_head(data_folder, seeg_sensors_files=[("SensorsInternal.h5", "")])
    # nearest_regions = head.compute_nearest_regions_to_sensors("SEEG", channels_inds)
    #
    # for ic, ch in enumerate(channels):
    #     print("\n" + ch + "<-" + str(nearest_regions[ic][1]) + str(nearest_regions[ic][2]))
    # channel_lbls = [u"G'1", u"G'2", u"G'11", u"G'12", u"M'6", u"M'7", u"L'6", u"L'7"]
    # channel_inds = [67, 68, 77, 78, 21, 22, 45, 46]
    # prepare_seeg_observable(os.path.join(SEEG_data, 'SZ1_0001.edf'), [10.0, 35.0], channels)
    # prepare_seeg_observable(os.path.join(SEEG_data, 'SZ2_0002.edf'), [15.0, 40.0], channels)
    # prepare_seeg_observable(os.path.join(SEEG_data, 'SZ5_0003.edf'), [20.0, 45.0], channels)
    EMPIRICAL = False
    if EMPIRICAL:
        main_fit_sim_hyplsa(p_name="ep_l_frontal_complex", data_folder=os.path.join(DATA_CUSTOM, 'Head'),
                            stats_model_name="vep_sde", EMPIRICAL=os.path.join(SEEG_data, 'SZ1_0001.edf'),
                            times_on_off=[10.0, 35.0], channel_lbls=channel_lbls, channel_inds=channel_inds,
                            fitmethod="optimizing", stan_service="CmdStan", results_dir=FOLDER_RES,
                            figure_dir=FOLDER_FIGURES)  # , stan_service="PyStan"
    else:
        main_fit_sim_hyplsa(p_name="ep_l_frontal_complex", data_folder=os.path.join(DATA_CUSTOM, 'Head'),
                            stats_model_name="vep_sde", fitmethod="optimizing", stan_service="CmdStan",
                            results_dir=FOLDER_RES, figure_dir=FOLDER_FIGURES) #, stan_service="PyStan"
