import os
from scipy.io import savemat, loadmat
import numpy as np

from tvb_epilepsy.base.constants.model_constants import K_DEF
from tvb_epilepsy.base.constants.model_inversion_constants import OBSERVATION_MODELS
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.service.model_inversion.stan.cmdstan_service import CmdStanService
from tvb_epilepsy.service.model_inversion.stan.pystan_service import PyStanService
from tvb_epilepsy.top.scripts.hypothesis_scripts import from_hypothesis_to_model_config_lsa
from tvb_epilepsy.top.scripts.simulation_scripts import from_model_configuration_to_simulation
from tvb_epilepsy.top.scripts.fitting_data_scripts import prepare_seeg_observable_from_mne_file


def set_model_config_LSA(head, hyp, reader, writer, plotter, config, K_unscaled=K_DEF):
    # --------------------------Model configuration and LSA-----------------------------------
    model_config_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_ModelConfig.h5")
    hyp_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_LSA.h5")
    if os.path.isfile(hyp_file) and os.path.isfile(model_config_file):
        # Read existing model configuration and LSA hypotheses...
        model_configuration = reader.read_model_configuration(model_config_file)
        lsa_hypothesis = reader.read_hypothesis(hyp_file)
    else:
        # ...or generate new ones
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

    return model_configuration, lsa_hypothesis


def load_empirical_data(ts_file):
    # Try to read a dictionary of previously written to file empirical data
    signals_ts_dict = loadmat(ts_file)
    time = signals_ts_dict["time"].flatten()
    sensors_inds = np.array(signals_ts_dict["sensors_inds"]).flatten().tolist()
    sensors_lbls = np.array(signals_ts_dict["sensors_lbls"]).flatten().tolist()
    signals_ts_dict.update({"time": time, "sensors_inds": sensors_inds, "sensors_lbls": sensors_lbls})
    savemat(ts_file, signals_ts_dict)
    return signals_ts_dict, sensors_inds, sensors_lbls


def proprocess_empirical_data(head, empirical_file, ts_file, sensors_inds, sensors_lbls, dynamical_model,
                              times_on_off, time_units, plotter):
    if len(sensors_lbls) == 0:
        sensors_lbls = head.get_sensors_id().labels
    signals, time, sensors_inds = \
        prepare_seeg_observable_from_mne_file(empirical_file, dynamical_model, times_on_off,
                                              sensors_lbls, sensors_inds, time_units=time_units,
                                              win_len_ratio=10, plotter=plotter)[:3]
    inds = np.argsort(sensors_inds)
    sensors_inds = np.array(sensors_inds)[inds].flatten().tolist()
    sensors_lbls = np.array(sensors_lbls).flatten().tolist()
    all_signals = np.zeros((signals.shape[0], len(sensors_lbls)))
    all_signals[:, sensors_inds] = signals[:, inds]
    signals = all_signals
    del all_signals
    signals_ts_dict = {"time": time.flatten(), "signals": signals,
                       "sensors_inds": sensors_inds, "sensors_lbls": sensors_lbls}
    savemat(ts_file, signals_ts_dict)
    return signals_ts_dict, sensors_inds, sensors_lbls


def set_empirical_data(head, hypname, model_inversion, empirical_file, sensors_inds, sensors_lbls,
                       dynamical_model, time_units, plotter, config):
    # ---------------------------------------Get empirical data-------------------------------------------
    model_inversion.target_data_type = "empirical"
    ts_file = os.path.join(config.out.FOLDER_RES, hypname + "_ts_empirical.mat")
    try:
        signals_ts_dict, sensors_inds, sensors_lbls = load_empirical_data(ts_file)
    except:
        # ... or preprocess empirical data for the first time:
        signals_ts_dict, sensors_inds, sensors_lbls = \
            proprocess_empirical_data(head, empirical_file, ts_file, sensors_inds, sensors_lbls,
                                      dynamical_model, time_units, plotter)
    return signals_ts_dict, sensors_inds, head.get_sensors().labels


def set_simulated_data(head, hypname, lsa_hypothesis, model_configuration, model_inversion, statistical_model,
                       sensors_inds, stan_model_name, dynamical_model, config):
    model_inversion.target_data_type = "simulated"
    ts_file = os.path.join(config.out.FOLDER_RES, hypname + "_ts.h5")
    signals_ts_dict = \
        from_model_configuration_to_simulation(model_configuration, head, lsa_hypothesis,
                                               sim_type="realistic", dynamical_model=dynamical_model,
                                               ts_file=ts_file, plot_flag=True, config=config)
    # if len(sensors_inds) > 1:  # get_bipolar_channels(sensors_inds, sensors_lbls)
    #     sensors_inds, sensors_lbls = head.get_sensors_id().get_bipolar_sensors(sensors_inds=sensors_inds)
    if statistical_model.observation_model.value in OBSERVATION_MODELS.SEEG.value:
        manual_selection = sensors_inds
        signals_labels = head.get_sensors_id().labels
    else:
        signals_labels = head.connectivity.region_labels
        if stan_model_name.find("vep-fe-rev") >= 0:
            manual_selection = statistical_model.active_regions
        else:
            manual_selection = []
    return signals_ts_dict, manual_selection, signals_labels


def plot_target_signals(signals_ts_dict, signals, time, signals_labels, hypname,
                        model_inversion, statistical_model, writer, plotter, config):
    if signals_ts_dict.get("signals", None) is not None:
        signals_ts_dict["signals"] -= signals_ts_dict["signals"].min()
        signals_ts_dict["signals"] /= signals_ts_dict["signals"].max()
        if statistical_model.observation_model == "seeg_logpower":
            special_idx = model_inversion.signals_inds
        else:
            special_idx = []
        plotter.plot_raster({'Target Signals': signals_ts_dict["signals"]}, signals_ts_dict["time"].flatten(),
                            time_units="ms", title=hypname + ' Target Signals raster',
                            special_idx=special_idx, offset=0.1, labels=signals_labels)
    plotter.plot_timeseries({'Target Signals': signals}, time, time_units="ms",
                            title=hypname + ' Target Signals',
                            labels=signals_labels[model_inversion.signals_inds])
    writer.write_model_inversion_service(model_inversion, os.path.join(config.out.FOLDER_RES,
                                                                       hypname + "_ModelInversionService.h5"))
    writer.write_generic(statistical_model, config.out.FOLDER_RES, hypname + "_StatsModel.h5")


def build_stan_service_and_model(stan_service, stan_model_name, fitmethod, config):
    # Compile or load model:
    # model_code_path = os.path.join(STATS_MODELS_PATH, stats_model_name + ".stan")
    model_code_path = os.path.join(config.generic.STATS_MODELS_PATH, stan_model_name + ".stan")
    if isequal_string(stan_service, "CmdStan"):
        stan_service = CmdStanService(model_name=stan_model_name, model=None, model_code=None,
                                      model_code_path=model_code_path,
                                      fitmethod=fitmethod, random_seed=12345, init="random", config=config)
    else:
        stan_service = PyStanService(model_name=stan_model_name, model=None, model_code=None,
                                     model_code_path=model_code_path,
                                     fitmethod=fitmethod, random_seed=12345, init="random", config=config)
    stan_service.set_or_compile_model()
    return stan_service


def plot_fitting_results(ests, samples, R_hat, stan_model_name, model_data, statistical_model, model_inversion,
                         model_configuration, lsa_hypothesis, plotter,
                         pair_plot_params=["time_scale", "k", "sigma", "epsilon", "amplitude", "offset"],
                         region_violin_params=["x0", "x_init", "z_init"],
                         x0_star_mu=None, x_init_mu=None, z_init_mu=None):

    simulation_values = {"x0": model_configuration.x0, "x1eq": model_configuration.x1EQ,
                         "x1init": model_configuration.x1EQ, "zinit": model_configuration.zEQ}

    if stan_model_name.find("vep-fe-rev") >= 0:
        input_signals_str = "seeg_log_power"
        # pair_plot_params = ["time_scale", "k", "sigma", "epsilon", "amplitude", "offset"],
        # region_violin_params = ["x0", "x_init", "z_init"]
        if model_inversion.target_data_type.find("empirical") >= 0:
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
        input_signals_str = "signals"
        # pair_plot_params = ["tau1", "tau0", "K", "sig_init", "sig", "eps", "scale_signal", "offset_signal"]
        # region_violin_params = ["x0", "x1eq", "x1init", "zinit"]
        connectivity_plot = False
        estMC = lambda est: est["MC"]
        region_mode = "all"
        if model_inversion.target_data_type.find("empirical") >= 0:
            priors = {"x0": model_inversion.x0[statistical_model.active_regions],
                      "x1eq": model_data["x1eq_max"]
                              - statistical_model.parameters["x1eq_star"].mean[statistical_model.active_regions],
                      "x_init": statistical_model.parameters["x1init"].mean[statistical_model.active_regions],
                      "z_init": statistical_model.parameters["zinit"].mean[statistical_model.active_regions]}
        else:
            priors = simulation_values
    plotter.plot_fit_results(model_inversion, ests, samples, statistical_model, model_data[input_signals_str],
                             R_hat, model_data["time"], priors, region_mode,
                             seizure_indices=lsa_hypothesis.get_regions_disease_indices(),
                             trajectories_plot=True, connectivity_plot=connectivity_plot,
                             pair_plot_params=pair_plot_params, region_violin_params=region_violin_params)