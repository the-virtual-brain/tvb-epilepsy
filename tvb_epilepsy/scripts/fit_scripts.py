# encoding=utf8

import os

import numpy as np

from tvb_epilepsy.base.constants.configurations import FOLDER_RES, DATA_CUSTOM, STATS_MODELS_PATH, FOLDER_VEP_HOME, \
                                                                                                         FOLDER_FIGURES
from tvb_epilepsy.base.constants.module_constants import TVB, CUSTOM
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, ensure_list, list_of_dicts_to_dicts_of_ndarrays
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.service.model_inversion.sde_model_inversion_service import \
    SDEModelInversionService
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
    head, hypos = from_head_to_hypotheses(ep_name, data_mode=CUSTOM, data_folder=os.path.join(DATA_CUSTOM, 'Head'),
                                          plot_head=False, figure_dir=figure_dir, logger=logger)

    for hyp in hypos: #hypotheses:
        # --------------------------Model configuration and LSA-----------------------------------
        model_configuration, lsa_hypothesis, model_configuration_service, lsa_service = \
            from_hypothesis_to_model_config_lsa(hyp, head, eigen_vectors_number=None, weighted_eigenvector_sum=True,
                                                plot_flag=True, figure_dir=figure_dir, logger=logger, K=10.0)

        dynamical_model = "EpileptorDP2D"
        if os.path.isfile(EMPIRICAL):
            # ---------------------------------------Get empirical data----------------------------------------------
            target_data_type = "empirical"
            observation, time, fs = prepare_seeg_observable(EMPIRICAL, times_on_off, channel_lbls, log_flag=True)
            vois_ts_dict = {"time": time, "signals": observation}

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

        # -------------------------- Get model_data and observation signals: -------------------------------------------
        model_inversion_service = SDEModelInversionService(model_configuration, lsa_hypothesis, head,
                                                            dynamical_model, sde_mode="x1z", logger=logger)
        statistical_model = model_inversion_service.generate_statistical_model(MC=os.path.join(FOLDER_VEP_HOME,
                                                                                                   "MC.h5"))
        if len(model_inversion_service.signals_inds) < head.get_sensors_id().number_of_sensors:
                statistical_model = \
                    model_inversion_service.update_active_regions_seeg(statistical_model)
        model_inversion_service.write_to_h5(FOLDER_RES, lsa_hypothesis.name + "_ModelInversionService.h5")
        statistical_model.write_to_h5(results_dir, lsa_hypothesis.name + "_StatsModel.h5")

        statistical_model = model_inversion_service.update_active_regions(statistical_model,
                                                                          methods=["e_values", "LSA"],
                                                                          active_regions_th=0.1, reset=True)
        signals, time, statistical_model = \
                model_inversion_service.set_target_data_and_time(target_data_type, vois_ts_dict, statistical_model,
                                                                 select_signals=True,
                                                                 power=True)  # rois=statistical_model.active_regions,
        try:
            model_data = stan_service.load_model_data_from_file()
        except:
            model_data = model_inversion_service.generate_model_data(statistical_model, signals)
            stan_service.write_model_data_to_file(model_data)

        # -------------------------- Fit and get estimates: ------------------------------------------------------------
        est, fit = stan_service.fit(model_data=model_data, debug=0, simulate=0, merge_outputs=False, **kwargs)
        convert_to_h5_model(est).write_to_h5(results_dir, lsa_hypothesis.name + "_fit_est.h5")
        est = ensure_list(est)
        for id_est, this_est in enumerate(est):
            model_inversion_service.plot_fit_results(this_est, statistical_model, signals, time=None,
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
    # VEP_HOME = os.path.join(USER_HOME, 'VEP/CC')
    # VEP_FOLDER = os.path.join(VEP_HOME, 'TVB3')
    # CT = os.path.join(VEP_FOLDER, 'CT')
    # SEEG = os.path.join(VEP_FOLDER, 'SEEG')
    # SEEG_data = os.path.join(VEP_FOLDER, 'SEEG_data')
    # channels = [u"G'1", u"G'2", u"G'3", u"G'8", u"G'9", u"G'10", u"G'11", u"G'12", u"M'6", u"M'7", u"M'8", u"L'4",
    #             u"L'5",  u"L'6", u"L'7", u"L'8", u"L'9"]
    # channel_inds = [67, 68, 69, 74, 75, 76, 77, 78, 21, 22, 23, 43, 44, 45, 46,  47, 48] #
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
    channels = [u"G'1", u"G'2", u"G'11", u"G'12", u"M'6", u"M'7", u"L'6", u"L'7"]
    channel_inds = [67, 68, 77, 78, 21, 22, 45, 46]
    # prepare_seeg_observable(os.path.join(SEEG_data, 'SZ1_0001.edf'), [10.0, 35.0], channels)
    # prepare_seeg_observable(os.path.join(SEEG_data, 'SZ2_0002.edf'), [15.0, 40.0], channels)
    # prepare_seeg_observable(os.path.join(SEEG_data, 'SZ5_0003.edf'), [20.0, 45.0], channels)
    stats_model_name = "vep_sde"
    # main_fit_sim_hyplsa(stats_model_name, EMPIRICAL=os.path.join(SEEG_data, 'SZ1_0001.edf'), times_on_off=[10.0, 35.0],
    #                    channel_lbls=channels, channel_inds=channel_inds)
    if len(channel_inds) > 1:
        channel_inds, _ = get_bipolar_channels(channel_inds, channels)
    main_fit_sim_hyplsa(p_name="ep_l_frontal_complex", data_folder=os.path.join(DATA_CUSTOM, 'Head'),
                        stats_model_name="vep_sde", EMPIRICAL="", times_on_off=[],
                        channel_lbls=channels, channel_inds=channel_inds,
                        fitmethod="optimizing", stan_service="CmdStan", results_dir=FOLDER_RES,
                        figure_dir=FOLDER_FIGURES) #, stan_service="PyStan"
