import os

from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.constants.model_constants import K_DEF
from tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.top.scripts.hypothesis_scripts import from_hypothesis_to_model_config_lsa
from tvb_epilepsy.top.scripts.simulation_scripts import from_model_configuration_to_simulation
from tvb_epilepsy.top.scripts.fitting_data_scripts import *


def set_model_config_LSA(head, hyp, reader, config, K_unscaled=K_DEF):
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
                                                config=config, K=K_unscaled, save_flag=True, plot_flag=True)
    return model_configuration, lsa_hypothesis


def set_empirical_data(empirical_file, ts_file, head, sensors_lbls, sensors_id=0, times_on_off=[],
                       label_strip_fun=None, plotter=False, **kwargs):
    try:
        return H5Reader().read_timeseries(ts_file)
    except:
        # ... or preprocess empirical data for the first time:
        if len(sensors_lbls) == 0:
            sensors_lbls = head.get_sensors_id(sensor_ids=sensors_id).labels
        signals = prepare_seeg_observable_from_mne_file(empirical_file, head.get_sensors_id(sensor_ids=sensors_id),
                                                        sensors_lbls, times_on_off, label_strip_fun=label_strip_fun,
                                                        bipolar=False, plotter=plotter, **kwargs)
        H5Writer().write_timeseries(signals, ts_file)
        return signals


def set_simulated_target_data(ts_file, model_configuration, head, lsa_hypothesis, statistical_model, sensors_id=0,
                              times_on_off=[], plotter=False, config=Config(), **kwargs):
    if statistical_model.observation_model == OBSERVATION_MODELS.SEEG_LOGPOWER.value:
        seeg_gain_mode = "exp"
    else:
        seeg_gain_mode = "lin"
    signals = from_model_configuration_to_simulation(model_configuration, head, lsa_hypothesis,
                                                     sim_type="paper", ts_file=ts_file,
                                                     seeg_gain_mode=seeg_gain_mode, config=config)
    if statistical_model.observation_model in OBSERVATION_MODELS.SEEG.value:
        if statistical_model.observation_model != OBSERVATION_MODELS.SEEG_LOGPOWER.value:
            try:
                signals = signals["seeg"][sensors_id]
            except:
                signals = TimeseriesService().compute_seeg(signals["source"].get_source(),
                                                           head.get_sensors_id(sensor_ids=sensors_id))[0]
        else:
            signals = TimeseriesService().compute_seeg(signals["source"].get_source(),
                                                       head.get_sensors_id(sensor_ids=sensors_id), sum_mode="exp")[0]

        signals = prepare_seeg_observable(signals, times_on_off, plotter=plotter, **kwargs)
    else:
        signals = prepare_signal_observable(signals["source"].get_source(), times_on_off, plotter=plotter, **kwargs)
    return signals


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