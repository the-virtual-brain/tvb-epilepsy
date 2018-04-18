import os

from tvb_epilepsy.base.constants.config import Config
from tvb_epilepsy.base.constants.model_constants import K_DEF
from tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list, generate_region_labels
from tvb_epilepsy.base.model.timeseries import TimeseriesDimensions, Timeseries
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



def samples_to_timeseries(samples, model_data, target_data=None, region_labels=[], region_mode="all"):
    samples = ensure_list(samples)

    if isinstance(target_data, Timeseries):
        time = target_data.time_line
        n_target_data = target_data.number_of_labels
        target_data_labels = target_data.space_labels
    else:
        time = model_data.get("time", False)
        n_target_data = samples[0]["fit_target_data"]
        target_data_labels = generate_region_labels(n_target_data, [], ". ", False)

    if time:
        time_start = time[0]
        time_step = np.diff(time).mean()
    else:
        time_start = 0
        time_step = 1

    if isinstance(target_data, Timeseries):
        target_data = Timeseries(target_data,
                                 {TimeseriesDimensions.SPACE.value: target_data_labels,
                                  TimeseriesDimensions.VARIABLES.value: ["target_data"]},
                                 time_start=time_start, time_step=time_step,
                                 time_unit=samples[0]["target_data"].time_unit)

    (n_samples, n_times, n_regions) = samples[0]["x1"]
    active_regions = model_data.get("active_regions", range(n_regions))
    regions_labels = generate_region_labels(n_regions, region_labels, ". ", False)
    if region_mode == "active" and n_regions > len(active_regions):
        regions_labels = regions_labels[active_regions]

    for sample in ensure_list(samples):
        for x in ["x1", "z", "dX1t", "dZt"]:
            try:
                sample[x] = Timeseries(sample[x].T, {TimeseriesDimensions.SPACE.value: regions_labels,
                                                     TimeseriesDimensions.VARIABLES.value: [x]},
                                       time_start=time_start, time_step=time_step, time_unit=target_data.time_unit)
            except:
                pass

        sample["fit_target_data"] = Timeseries(sample["fit_target_data"].T,
                                               {TimeseriesDimensions.SPACE.value: target_data_labels,
                                                TimeseriesDimensions.VARIABLES.value: ["fit_target_data"]},
                               time_start=time_start, time_step=time_step)

    return samples, target_data
