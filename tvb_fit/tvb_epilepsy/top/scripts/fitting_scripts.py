import os

import numpy as np

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.constants.model_constants import K_UNSCALED_DEF, TAU1_DEF, TAU0_DEF
from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants import OBSERVATION_MODELS, SEIZURE_LENGTH, \
    HIGH_HPF, LOW_HPF, LOW_LPF, HIGH_LPF, WIN_LEN_RATIO, BIPOLAR, TARGET_DATA_PREPROCESSING
from tvb_fit.base.utils.log_error_utils import initialize_logger
from tvb_fit.base.utils.data_structures_utils import ensure_list, generate_region_labels
from tvb_fit.base.utils.file_utils import move_overwrite_files_to_folder_with_wildcard
from tvb_fit.service.timeseries_service import TimeseriesService
from tvb_fit.tvb_epilepsy.base.model.timeseries import TimeseriesDimensions, Timeseries
from tvb_fit.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_fit.tvb_epilepsy.top.scripts.hypothesis_scripts import from_hypothesis_to_model_config_lsa
from tvb_fit.tvb_epilepsy.top.scripts.pse_scripts import pse_from_lsa_hypothesis
from tvb_fit.tvb_epilepsy.top.scripts.simulation_scripts import from_model_configuration_to_simulation
from tvb_fit.tvb_epilepsy.top.scripts.fitting_data_scripts import prepare_seeg_observable_from_mne_file, \
    prepare_simulated_seeg_observable, prepare_signal_observable
from tvb_fit.tvb_epilepsy.service.lsa_service import LSAService
from tvb_fit.tvb_epilepsy.io.h5_writer import H5Writer
from tvb_fit.tvb_epilepsy.io.h5_reader import H5Reader

logger = initialize_logger(__name__)


def set_model_config_LSA(head, hyp, reader, config, K_unscaled=K_UNSCALED_DEF, tau1=TAU1_DEF, tau0=TAU0_DEF,
                         pse_flag=True, plotter=None, writer=None):
    model_configuration_builder = None
    lsa_service = None
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
                                                config=config, K_unscaled=K_unscaled, tau1=tau1, tau0=tau0,
                                                save_flag=True, plot_flag=True)
        # --------------Parameter Search Exploration (PSE)-------------------------------
    if pse_flag:
        psa_lsa_file = os.path.join(config.out.FOLDER_RES, hyp.name + "_PSE_LSA_results.h5")
        try:
            pse_results = reader.read_dictionary(psa_lsa_file)
        except:
            logger.info("\n\nRunning PSE LSA...")
            n_samples = 100
            all_regions_indices = np.array(range(head.number_of_regions))
            disease_indices = lsa_hypothesis.regions_disease_indices
            healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()
            if model_configuration_builder is None:
                model_configuration_builder = \
                    ModelConfigurationBuilder("EpileptorDP2D", head.connectivity, K_unscaled=K_unscaled)
            if lsa_service is None:
                lsa_service =  LSAService(eigen_vectors_number=None, weighted_eigenvector_sum=True)
            pse_results = \
                pse_from_lsa_hypothesis(n_samples, lsa_hypothesis, head.connectivity.normalized_weights,
                                        model_configuration_builder, lsa_service, head.connectivity.region_labels,
                                        param_range=0.1, global_coupling=[{"indices": all_regions_indices}],
                                        healthy_regions_parameters=[ {"name": "x0_values", "indices": healthy_indices}],
                                        logger=logger, save_flag=False)[0]
            if plotter:
                plotter.plot_lsa(lsa_hypothesis, model_configuration, lsa_service.weighted_eigenvector_sum,
                                  lsa_service.eigen_vectors_number, head.connectivity.region_labels, pse_results)
            if writer:
                writer.write_dictionary(pse_results, psa_lsa_file)
    else:
        pse_results = {}
    return model_configuration, lsa_hypothesis, pse_results


def set_empirical_data(empirical_file, ts_file, head, sensors_lbls, sensor_id=0, seizure_length=SEIZURE_LENGTH,
                       times_on_off=[], time_units="ms", label_strip_fun=None,
                       preprocessing=TARGET_DATA_PREPROCESSING, low_hpf=LOW_HPF, high_hpf=HIGH_HPF, low_lpf=LOW_LPF,
                       high_lpf=HIGH_LPF, bipolar=BIPOLAR, win_len_ratio=WIN_LEN_RATIO,
                       plotter=None, title_prefix=""):
    try:
        return H5Reader().read_timeseries(ts_file)
    except:
        seizure_name = os.path.basename(empirical_file).split(".")[0]
        if title_prefix.find(seizure_name) < 0:
            title_prefix = title_prefix + seizure_name
        # ... or preprocess empirical data for the first time:
        if len(sensors_lbls) == 0:
            sensors_lbls = head.get_sensors_by_index(sensor_ids=sensor_id).labels
        signals = prepare_seeg_observable_from_mne_file(empirical_file, head.get_sensors_by_index(sensor_ids=sensor_id),
                                                        sensors_lbls, seizure_length, times_on_off, time_units,
                                                        label_strip_fun, preprocessing,
                                                        low_hpf, high_hpf, low_lpf, high_lpf,
                                                        bipolar, win_len_ratio, plotter, title_prefix)
        H5Writer().write_timeseries(signals, ts_file)
    move_overwrite_files_to_folder_with_wildcard(os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              "fitData_EmpiricalSEEG"),
                                                 os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              title_prefix.replace(" ", "_")) + "*")
    return signals


def set_multiple_empirical_data(empirical_files, ts_file, head, sensors_lbls, sensor_id=0,
                                seizure_length=SEIZURE_LENGTH, times_on=[], time_length=25600, time_units="ms",
                                label_strip_fun=None, preprocessing=TARGET_DATA_PREPROCESSING,
                                low_hpf=LOW_HPF, high_hpf=HIGH_HPF, low_lpf=LOW_LPF, high_lpf=HIGH_LPF,
                                bipolar=BIPOLAR, win_len_ratio=WIN_LEN_RATIO, plotter=None, title_prefix=""):
    empirical_files = ensure_list(ensure_list(empirical_files))
    n_seizures = len(empirical_files)
    times_on = ensure_list(times_on)
    signals = []
    ts_filename = ts_file.split(".h5")[0]
    for empirical_file, time_on in zip(empirical_files, times_on):
        seizure_name = os.path.basename(empirical_file).split(".")[0]
        signals.append(set_empirical_data(empirical_file, "_".join([ts_filename, seizure_name]) + ".h5",
                                          head, sensors_lbls, sensor_id, seizure_length,
                                          [time_on, time_on + time_length], time_units,
                                          label_strip_fun,preprocessing, low_hpf, high_hpf, low_lpf, high_lpf,
                                          bipolar, win_len_ratio, plotter, title_prefix))
    if n_seizures > 1:
        signals = TimeseriesService().concatenate_in_time(signals)
    else:
        signals = signals[0]
    if plotter:
        title_prefix = title_prefix + "MultiseizureEmpiricalSEEG"
        plotter.plot_raster({"ObservationRaster": signals.squeezed}, signals.time_line, time_units=signals.time_unit,
                            special_idx=[], offset=0.1, title='Multiseizure Observation Raster Plot',
                            figure_name=title_prefix + 'ObservationRasterPlot', labels=signals.space_labels)
        plotter.plot_timeseries({"Observation": signals.squeezed}, signals.time_line, time_units=signals.time_unit,
                                special_idx=[], title='Observation Time Series',
                                figure_name=title_prefix + 'ObservationTimeSeries', labels=signals.space_labels)
    move_overwrite_files_to_folder_with_wildcard(os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              "fitData_EmpiricalSEEG"),
                                                 os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              title_prefix.replace(" ", "_")) + "*")
    return signals, n_seizures


def set_simulated_target_data(ts_file, head, lsa_hypothesis, probabilistic_model, sensor_id=0,
                              sim_type="paper", times_on_off=[], config=Config(),
                              preprocessing=TARGET_DATA_PREPROCESSING,
                              low_hpf=LOW_HPF, high_hpf=HIGH_HPF, low_lpf=LOW_LPF, high_lpf=HIGH_LPF,
                              bipolar=BIPOLAR, win_len_ratio=WIN_LEN_RATIO, plotter=None, title_prefix=""):
    signals, simulator = from_model_configuration_to_simulation(probabilistic_model.model_config,
                                                                head, lsa_hypothesis,
                                                                sim_type=sim_type, ts_file=ts_file,
                                                                config=config, plotter=plotter)
    try:
        probabilistic_model.ground_truth.update({"tau1": np.mean(simulator.model.tau1),
                                                 "tau0": np.mean(simulator.model.tau0),
                                                 "sigma": np.mean(simulator.settings.noise_intensity)})
    except:
        probabilistic_model.ground_truth.update({"tau1": np.mean(simulator.model.tt),
                                                 "tau0": 1.0 / np.mean(simulator.model.r),
                                                 "sigma": np.mean(simulator.settings.noise_intensity)})
    x1z = signals["source"].get_time_window_by_units(times_on_off[0], times_on_off[1])
    x1 = x1z.x1.squeezed
    z = x1z.z.squeezed
    del x1z
    probabilistic_model.ground_truth.update({"x1_init": x1[0].squeeze(), "z_init": z[0].squeeze(),})
    del x1, z
    signals = signals["source"].get_source()
    signals.data = -signals.data  # change sign to fit x1

    if probabilistic_model.observation_model in OBSERVATION_MODELS.SEEG.value:
        log_flag = probabilistic_model.observation_model == OBSERVATION_MODELS.SEEG_LOGPOWER.value
        title_prefix = title_prefix + "SimSEEG"
        signals = prepare_simulated_seeg_observable(signals, head.get_sensors_by_index(sensor_ids=sensor_id),
                                                    probabilistic_model.time_length, log_flag, times_on_off, [],
                                                    preprocessing, low_hpf, high_hpf, low_lpf, high_lpf, bipolar,
                                                    win_len_ratio, plotter, title_prefix)

    else:
        title_prefix = title_prefix + "SimSource"
        signals = prepare_signal_observable(signals, probabilistic_model.time_length, times_on_off, [],
                                            preprocessing, low_hpf, high_hpf, low_lpf, high_lpf,
                                            win_len_ratio, plotter, title_prefix)
    move_overwrite_files_to_folder_with_wildcard(os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              "fitData_" + title_prefix),
                                                 os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              title_prefix.replace(" ", "_")) + "*")
    move_overwrite_files_to_folder_with_wildcard(os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              "Simulation"),
                                                 os.path.join(plotter.config.out.FOLDER_FIGURES, "Simulated*"))
    return signals, simulator


def samples_to_timeseries(samples, model_data, target_data=None, region_labels=[]):
    samples = ensure_list(samples)

    if isinstance(target_data, Timeseries):
        time = target_data.time_line
        n_target_data = target_data.number_of_labels
        target_data_labels = target_data.space_labels
    else:
        time = model_data.get("time", False)
        n_target_data = samples[0]["fit_target_data"]
        target_data_labels = generate_region_labels(n_target_data, [], ". ", False)

    if time is not False:
        time_start = time[0]
        time_step = np.diff(time).mean()
    else:
        time_start = 0
        time_step = 1

    if not isinstance(target_data, Timeseries):
        target_data = Timeseries(target_data,
                                 {TimeseriesDimensions.SPACE.value: target_data_labels,
                                  TimeseriesDimensions.VARIABLES.value: ["target_data"]},
                                  time_start=time_start, time_step=time_step)

    (n_times, n_regions, n_samples) = samples[0]["x1"].T.shape
    active_regions = model_data.get("active_regions", np.array(range(n_regions)))
    region_labels = generate_region_labels(np.maximum(n_regions, len(region_labels)), region_labels, ". ", False)
    if len(region_labels) > len(active_regions):
        region_labels = region_labels[active_regions]

    x1 = np.empty((n_times, n_regions, 0))
    for sample in ensure_list(samples):
        for x in ["x1", "z", "x1_star", "z_star", "dX1t", "dZt", "dWt", "dX1t_star", "dZt_star", "dWt_star"]:
            try:
                if x == "x1":
                    x1 = np.concatenate([x1, sample[x].T], axis=2)
                sample[x] = Timeseries(np.expand_dims(sample[x].T, 2), {TimeseriesDimensions.SPACE.value: region_labels,
                                                     TimeseriesDimensions.VARIABLES.value: [x]},
                                       time_start=time_start, time_step=time_step, time_unit=target_data.time_unit)

            except:
                pass

        sample["fit_target_data"] = Timeseries(np.expand_dims(sample["fit_target_data"].T, 2),
                                               {TimeseriesDimensions.SPACE.value: target_data_labels,
                                                TimeseriesDimensions.VARIABLES.value: ["fit_target_data"]},
                               time_start=time_start, time_step=time_step)

    return samples, target_data, np.nanmean(x1, axis=2).squeeze(), np.nanstd(x1, axis=2).squeeze()


def get_x1_estimates_from_samples(samples, model_data, region_labels=[], time_unit="ms"):
    time = model_data.get("time", False)
    if time is not False:
        time_start = time[0]
        time_step = np.diff(time).mean()
    else:
        time_start = 0
        time_step = 1
    if isinstance(samples[0]["x1"], np.ndarray):
        get_x1 = lambda x1: x1.T
    else:
        get_x1 = lambda x1: x1.squeezed
    (n_times, n_regions, n_samples) = get_x1(samples[0]["x1"]).shape
    active_regions = model_data.get("active_regions", np.array(range(n_regions)))
    region_labels = generate_region_labels(np.maximum(n_regions, len(region_labels)), region_labels, ". ", False)
    if len(region_labels) > len(active_regions):
        region_labels = region_labels[active_regions]
    x1 = np.empty((n_times, n_regions, 0))
    for sample in ensure_list(samples):
        x1 = np.concatenate([x1, get_x1(sample["x1"])], axis=2)
    x1_mean = Timeseries(np.nanmean(x1, axis=2).squeeze(), {TimeseriesDimensions.SPACE.value: region_labels,
                                                      TimeseriesDimensions.VARIABLES.value: ["x1"]},
                         time_start=time_start, time_step=time_step, time_unit=time_unit)
    x1_std = Timeseries(np.nanstd(x1, axis=2).squeeze(), {TimeseriesDimensions.SPACE.value: region_labels,
                                                            TimeseriesDimensions.VARIABLES.value: ["x1std"]},
                         time_start=time_start, time_step=time_step, time_unit=time_unit)
    return x1_mean, x1_std