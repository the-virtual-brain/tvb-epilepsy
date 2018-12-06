import os

import numpy as np

from tvb_fit.base.constants import Target_Data_Type
from tvb_fit.base.utils.log_error_utils import initialize_logger, warning
from tvb_fit.base.utils.data_structures_utils import ensure_list, generate_region_labels
from tvb_fit.base.utils.file_utils import move_overwrite_files_to_folder_with_wildcard
from tvb_fit.base.computations.math_utils import select_greater_values_array_inds
from tvb_fit.service.timeseries_service import TimeseriesService
from tvb_fit.samplers.stan.cmdstan_interface import CmdStanInterface
from tvb_fit.plot.head_plotter import HeadPlotter

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.constants.model_constants import K_UNSCALED_DEF, TAU1_DEF, TAU0_DEF
from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants import OBSERVATION_MODELS, SEIZURE_LENGTH, \
    HIGH_HPF, LOW_HPF, LOW_LPF, HIGH_LPF, WIN_LEN_RATIO, BIPOLAR, TARGET_DATA_PREPROCESSING, XModes, \
    compute_upsample, compute_seizure_length
from tvb_fit.tvb_epilepsy.base.model.timeseries import TimeseriesDimensions, Timeseries
from tvb_fit.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_fit.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_fit.tvb_epilepsy.service.lsa_service import LSAService
from tvb_fit.tvb_epilepsy.service.probabilistic_models_builders import SDEProbabilisticModelBuilder
from tvb_fit.tvb_epilepsy.top.scripts.hypothesis_scripts import from_hypothesis_to_model_config_lsa
from tvb_fit.tvb_epilepsy.top.scripts.pse_scripts import pse_from_lsa_hypothesis
from tvb_fit.tvb_epilepsy.top.scripts.simulation_scripts import from_model_configuration_to_simulation
from tvb_fit.tvb_epilepsy.top.scripts.fitting_data_scripts import prepare_seeg_observable_from_mne_file, \
    prepare_simulated_seeg_observable, prepare_signal_observable
from tvb_fit.tvb_epilepsy.io.h5_writer import H5Writer
from tvb_fit.tvb_epilepsy.io.h5_reader import H5Reader


logger = initialize_logger(__name__)


def path(name, base_path):
    return os.path.join(base_path , name + ".h5")


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


def get_2D_simulation(model_configuration, head, lsa_hypothesis, source2D_file, sim_times_on_off,
                      config=None, reader=None, writer=None, plotter=None):

    # --------------------- Get prototypical simulated data (simulate if necessary) --------------------------------
    try:
        source2D_ts = reader.read_timeseries(source2D_file)
    except:

        source2D_ts = from_model_configuration_to_simulation(model_configuration, head, lsa_hypothesis,
                                                             rescale_x1eq=-1.2, sim_type="fitting",
                                                             ts_file=source2D_file, config=config,
                                                             plotter=plotter, title_prefix="Source2D")[0]["source"]. \
            get_time_window_by_units(sim_times_on_off[0], sim_times_on_off[1])

    if config:
        move_overwrite_files_to_folder_with_wildcard(os.path.join(config.out.FOLDER_FIGURES, "Simulation"),
                                                     plotter.config.out.FOLDER_FIGURES + "/*Simulated*")
    if writer:
        writer.write_timeseries(source2D_ts, source2D_file)
    return source2D_ts


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
        if len(times_on_off) == 2:
            seizure_duration = np.diff(times_on_off)
        else:
            seizure_duration = times_on_off[0]
        signals = prepare_seeg_observable_from_mne_file(empirical_file, head.get_sensors_by_index(sensor_ids=sensor_id),
                                                        sensors_lbls, seizure_length, times_on_off, time_units,
                                                        label_strip_fun, preprocessing,
                                                        low_hpf, high_hpf, low_lpf, high_lpf,
                                                        bipolar, seizure_duration / win_len_ratio,
                                                        plotter, title_prefix)
        H5Writer().write_timeseries(signals, ts_file)
    move_overwrite_files_to_folder_with_wildcard(os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              "fitData_EmpiricalSEEG"),
                                                 os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              title_prefix.replace(" ", "_")) + "*")
    return signals


def set_multiple_empirical_data(empirical_files, ts_file, head, sensors_lbls, sensor_id=0,
                                seizure_length=SEIZURE_LENGTH, times_on=[], time_length=32000.0, time_units="ms",
                                label_strip_fun=None, preprocessing=TARGET_DATA_PREPROCESSING,
                                low_hpf=LOW_HPF, high_hpf=HIGH_HPF, low_lpf=LOW_LPF, high_lpf=HIGH_LPF,
                                bipolar=BIPOLAR, win_len_ratio=WIN_LEN_RATIO, plotter=None, title_prefix=""):
    empirical_files = ensure_list(ensure_list(empirical_files))
    n_seizures = len(empirical_files)
    times_on = ensure_list(times_on)
    if len(times_on) == n_seizures:
        times_on_off = []
        for time_on in times_on:
            times_on_off.append([time_on, time_on + time_length])
    else:
        times_on_off = n_seizures * [[time_length]]
    signals = []
    ts_filename = ts_file.split(".h5")[0]
    for empirical_file, time_on_off in zip(empirical_files, times_on_off):
        seizure_name = os.path.basename(empirical_file).split(".")[0]
        signals.append(set_empirical_data(empirical_file, "_".join([ts_filename, seizure_name]) + ".h5",
                                          head, sensors_lbls, sensor_id, seizure_length, time_on_off, time_units,
                                          label_strip_fun,preprocessing, low_hpf, high_hpf, low_lpf, high_lpf,
                                          bipolar, win_len_ratio, plotter, title_prefix))
    if n_seizures > 1:
        # Concatenate only the labels that exist in all signals:
        labels = signals[0].space_labels
        for signal in signals[1:]:
            labels = np.intersect1d(labels, signal.space_labels)
        signals = TimeseriesService().concatenate_in_time(signals, labels)
    else:
        signals = signals[0]
    if plotter:
        title_prefix = title_prefix + "MultiseizureEmpiricalSEEG"
        plotter.plot_raster({"ObservationRaster": signals.squeezed}, signals.time, time_units=signals.time_unit,
                            special_idx=[], offset=0.1, title='Multiseizure Observation Raster Plot',
                            figure_name=title_prefix + 'ObservationRasterPlot', labels=signals.space_labels)
        plotter.plot_timeseries({"Observation": signals.squeezed}, signals.time, time_units=signals.time_unit,
                                special_idx=[], title='Observation Time Series',
                                figure_name=title_prefix + 'ObservationTimeSeries', labels=signals.space_labels)
    move_overwrite_files_to_folder_with_wildcard(os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              "fitData_EmpiricalSEEG"),
                                                 os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              title_prefix.replace(" ", "_")) + "*")
    return signals, n_seizures


def set_simulated_target_data(ts_file, head, lsa_hypothesis, probabilistic_model, sensor_id=0,
                              rescale_x1eq=None, sim_type="paper", times_on_off=[], seizure_length=SEIZURE_LENGTH,
                              preprocessing=TARGET_DATA_PREPROCESSING, low_hpf=LOW_HPF, high_hpf=HIGH_HPF,
                              low_lpf=LOW_LPF, high_lpf=HIGH_LPF, bipolar=BIPOLAR, win_len_ratio=WIN_LEN_RATIO,
                              plotter=None, config=Config(), title_prefix=""):
    signals, simulator = from_model_configuration_to_simulation(probabilistic_model.model_config,
                                                                head, lsa_hypothesis, rescale_x1eq=rescale_x1eq,
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
    win_len = np.diff(times_on_off)[0]/win_len_ratio
    if probabilistic_model.observation_model in OBSERVATION_MODELS.SEEG.value:
        log_flag = probabilistic_model.observation_model == OBSERVATION_MODELS.SEEG_LOGPOWER.value
        # get timeseries to be positive
        title_prefix = title_prefix + "SimSEEG"
        signals = prepare_simulated_seeg_observable(signals, head.get_sensors_by_index(sensor_ids=sensor_id),
                                                    seizure_length, log_flag, times_on_off, [],
                                                    preprocessing, low_hpf, high_hpf, low_lpf, high_lpf, bipolar,
                                                    win_len, plotter, title_prefix)

    else:
        title_prefix = title_prefix + "SimSource"
        signals = prepare_signal_observable(signals, probabilistic_model.time_length, times_on_off, [],
                                            preprocessing, low_hpf, high_hpf, low_lpf, high_lpf,
                                            win_len, plotter, title_prefix)
    move_overwrite_files_to_folder_with_wildcard(os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              "fitData_" + title_prefix),
                                                 os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              title_prefix.replace(" ", "_")) + "*")
    move_overwrite_files_to_folder_with_wildcard(os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              "Simulation"),
                                                 os.path.join(plotter.config.out.FOLDER_FIGURES, "*Simulated*"))
    return signals, simulator


def get_target_timeseries(probabilistic_model, head, hypothesis, times_on, time_length, sensors_lbls, sensor_id,
                          observation_model, sim_target_file, empirical_target_file, sim_source_type="paper",
                          downsampling=1, preprocessing=[], empirical_files=[], config=Config(), plotter=None):

    # Some scripts for settting and preprocessing target signals:
    simulator = None
    log_flag = observation_model == OBSERVATION_MODELS.SEEG_LOGPOWER.value
    empirical_files = ensure_list(empirical_files)
    times_on = ensure_list(times_on)
    seizure_length = int(np.ceil(compute_seizure_length(probabilistic_model.tau0) / downsampling))
    if len(empirical_files) > 0:
        if len(preprocessing) == 0:
            if log_flag:
                preprocessing = ["hpf", "abs-envelope", "convolve", "decimate", "log"]# ["spectrogram", "log"] #
            else:
                preprocessing  = ["hpf", "mean_center", "abs-envelope", "convolve", "decimate"]
        # -------------------------- Get empirical data (preprocess edf if necessary) --------------------------
        signals, probabilistic_model.number_of_seizures = \
            set_multiple_empirical_data(empirical_files, empirical_target_file, head, sensors_lbls, sensor_id,
                                        seizure_length, times_on, time_length,
                                        label_strip_fun=lambda s: s.split("POL ")[-1], preprocessing=preprocessing,
                                        plotter=plotter, title_prefix="")
    else:
        probabilistic_model.number_of_seizures = 1
        # --------------------- Get fitting target simulated data (simulate if necessary) ----------------------
        probabilistic_model.target_data_type = Target_Data_Type.SYNTHETIC.value
        if len(preprocessing) == 0:
            preprocessin = []
            if sim_source_type == "paper":
                preprocessing = ["convolve"] # ["spectrogram", "log"]  #, "convolve" # ["hpf", "mean_center", "abs_envelope", "log"]
        if probabilistic_model.observation_model in OBSERVATION_MODELS.SEEG.value:
            preprocessing += ["mean_center"]
        preprocessing += ["decimate"]
        rescale_x1eq = -1.225
        if np.max(probabilistic_model.model_config.x1eq) > rescale_x1eq:
            rescale_x1eq = False
        signals, simulator = \
            set_simulated_target_data(sim_target_file, head, hypothesis, probabilistic_model, sensor_id,
                                      rescale_x1eq=rescale_x1eq, sim_type=sim_source_type,
                                      times_on_off=[times_on[0], times_on[0] + time_length],
                                      seizure_length=seizure_length,
                                      # Maybe change some of those for Epileptor 6D simulations:
                                      bipolar=False, preprocessing=preprocessing,
                                      plotter=plotter, config=config, title_prefix="")
    return signals, probabilistic_model, simulator


def set_target_timeseries(probabilistic_model, model_inversion, signals, sensors, head,
                          target_data_file="", writer=None, plotter=None):

    # -------------------------- Select and set target data from signals ---------------------------------------
    if probabilistic_model.observation_model in OBSERVATION_MODELS.SEEG.value:
        model_inversion.auto_selection = "rois-power"  # -rois
        model_inversion.sensors_per_electrode = 2
    target_data, probabilistic_model = \
        model_inversion.set_target_data_and_time(signals, probabilistic_model, head=head, sensors=sensors)

    if plotter:
        plotter.plot_raster({'Target Signals': target_data.squeezed}, target_data.time,
                            time_units=target_data.time_unit, title='Fit-Target Signals raster',
                            offset=0.1, labels=target_data.space_labels)
        plotter.plot_timeseries({'Target Signals': target_data.squeezed}, target_data.time,
                                time_units=target_data.time_unit,
                                title='Fit-Target Signals', labels=target_data.space_labels)

        HeadPlotter(plotter.config)._plot_gain_matrix(sensors, head.connectivity.region_labels,
                                                      title="Active regions -> target data projection",
                                                      show_x_labels=True, show_y_labels=True,
                                                      x_ticks=sensors. \
                                                         get_sensors_inds_by_sensors_labels(target_data.space_labels),
                                                      y_ticks=probabilistic_model.active_regions)

    if writer:
        writer.write_timeseries(target_data, target_data_file)

    return target_data, probabilistic_model, model_inversion


def set_prior_parameters(probabilistic_model, target_data, source2D_ts, x1prior_ts, problstc_model_file,
                        probabilistic_model_builder=SDEProbabilisticModelBuilder,
                         params_names=[XModes.X0MODE.value, "sigma_" + XModes.X0MODE.value,
                                       "x1_init", "z_init", "tau1",  # "tau0", "K", "x1",
                                     "sigma", "dWt", "epsilon", "scale", "offset"], normal_flag=False,
                         step_prefix="", writer=None, plotter=None):

    # ---------------------------------Finally set priors for the parameters-------------------------------------
    probabilistic_model.time_length = target_data.time_length
    probabilistic_model.upsample = \
        compute_upsample(probabilistic_model.time_length / probabilistic_model.number_of_seizures,
                         compute_seizure_length(probabilistic_model.tau0), probabilistic_model.tau0)

    probabilistic_model.parameters = probabilistic_model_builder(probabilistic_model). \
        generate_parameters(params_names, probabilistic_model.parameters, target_data, source2D_ts, x1prior_ts)

    if plotter:
        plotter.plot_probabilistic_model(probabilistic_model, step_prefix + "Probabilistic Model")
    if writer:
        writer. \
            write_probabilistic_model(probabilistic_model, probabilistic_model.model_config.number_of_regions,
                                      problstc_model_file)
    return probabilistic_model


def run_fitting(probabilistic_model, stan_model_name, model_data, target_data, config, head=None, seizure_indices=[],
                pair_plot_params=["tau1", "tau0", "K", "sigma", "epsilon", "scale", "offset"],
                region_violin_params=["x0", "PZ", "x1eq", "zeq"], state_variables=["x1", "z"],
                state_noise_variables=["dWt", "dX1t", "dZt"], fit_flag=True, test_flag=False, base_path="",
                fitmethod="sample", n_chains_or_runs=2, output_samples=200, num_warmup=100, min_samples_per_chain=200,
                max_depth=15, delta=0.95, iter=500000, tol_rel_obj=1e-6, debug=1, simulate=0,
                step_prefix='', writer=None, plotter=None, **kwargs):
    # ------------------------------Stan model and service--------------------------------------
    model_code_path = os.path.join(config.generic.PROBLSTC_MODELS_PATH, stan_model_name + ".stan")
    stan_interface = CmdStanInterface(model_name=stan_model_name, model_dir=base_path,
                                      model_code_path=model_code_path, fitmethod=fitmethod, config=config)
    stan_interface.model_data_path = os.path.join(base_path, "ModelData.h5")

    # -------------------------- Fit and get estimates: ------------------------------------------------------------
    n_chains_or_runs = np.where(test_flag, 2, n_chains_or_runs)
    output_samples = np.where(test_flag, 20, max(int(np.round(output_samples * 1.0 / n_chains_or_runs)),
                                                 min_samples_per_chain))

    # Sampling (HMC)
    num_samples = output_samples
    num_warmup = np.where(test_flag, 30, num_warmup)
    max_depth = np.where(test_flag, 7, max_depth)
    delta = np.where(test_flag, 0.8, delta)
    # ADVI or optimization:
    iter = np.where(test_flag, 10000, iter)
    if fitmethod.find("sampl") >= 0:
        skip_samples = num_warmup
    else:
        skip_samples = 0
    prob_model_name = probabilistic_model.name.split(".")[0]
    if fit_flag == "prepare":
        stan_interface.prepare_fit(debug=debug, simulate=simulate, model_data=model_data,
                                   n_chains_or_runs=n_chains_or_runs, refresh=1, iter=iter, tol_rel_obj=tol_rel_obj,
                                   output_samples=output_samples, num_warmup=num_warmup, num_samples=num_samples,
                                   max_depth=max_depth, delta=delta, save_warmup=1, output_path=base_path,
                                   **kwargs)
    else:
        if fit_flag == "fit":
            stan_interface.set_or_compile_model()
            estimates, samples, summary = stan_interface.fit(debug=debug, simulate=simulate, model_data=model_data,
                                                             n_chains_or_runs=n_chains_or_runs, refresh=1,
                                                             iter=iter, tol_rel_obj=tol_rel_obj,
                                                             output_samples=output_samples,
                                                             num_warmup=num_warmup, num_samples=num_samples,
                                                             max_depth=max_depth, delta=delta,
                                                             save_warmup=1, plot_warmup=1, output_path=base_path,
                                                             **kwargs)
            # TODO: check if write_dictionary is enough for estimates, samples, summary and info_crit
            if writer:
                writer.write_list_of_dictionaries(estimates, path(prob_model_name + "_FitEst", base_path))
                # writer.write_list_of_dictionaries(samples, path(prob_model_name + "_FitSamples", base_path))
                if summary is not None:
                    writer.write_dictionary(summary, path(prob_model_name + "_FitSummary", base_path))
        else:
            stan_interface.set_output_files(base_path=base_path, update=True)
            estimates, samples, summary = stan_interface.read_output()

        # Model comparison:
        info_crit = \
            stan_interface.compute_information_criteria(samples, None, skip_samples=skip_samples,
                                                        # parameters=["amplitude_star", "offset_star", "epsilon_star",
                                                        #                  "sigma_star", "time_scale_star", "x0_star",
                                                        #                  "x_init_star", "z_init_star", "z_eta_star"],
                                                        merge_chains_or_runs_flag=False)

        if writer:
            writer.write_dictionary(info_crit, path(prob_model_name + "_InfoCrit", base_path))

        # Interface backwards with INS stan models
        # from tvb_fit.service.model_inversion.vep_stan_dict_builder import convert_params_names_from_ins
        # estimates, samples, Rhat, model_data = \
        #     convert_params_names_from_ins([estimates, samples, Rhat, model_data])
        if fitmethod.find("opt") < 0:
            stats = stan_interface.get_summary_stats(summary, ["R_hat", "N_Eff/s"])
        else:
            stats = None

        if plotter:
            # -------------------------- Plot fitting results: ------------------------------------------------------------
            try:
                if fitmethod.find("sampl") >= 0:
                    plotter.plot_HMC(samples, figure_name=step_prefix + prob_model_name + " HMC NUTS trace")

                plotter.plot_fit_results(estimates, samples, model_data, target_data, probabilistic_model, info_crit,
                                         stats=stats, seizure_indices=seizure_indices,
                                         pair_plot_params=pair_plot_params, region_violin_params=region_violin_params,
                                         state_variables=state_variables, state_noise_variables=state_noise_variables,
                                         region_labels=head.connectivity.region_labels, skip_samples=skip_samples,
                                         title_prefix=step_prefix + prob_model_name)
            except:
                warning("Fitting plotting failed for step %s" % step_prefix)

        return estimates, samples, summary, info_crit


def samples_to_timeseries(samples, model_data, target_data=None, region_labels=[]):
    samples = ensure_list(samples)

    if isinstance(target_data, Timeseries):
        time = target_data.time
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


def get_x1_estimates_from_samples(samples, time=None, active_regions=[], region_labels=[], time_unit="ms"):
    if time is not None:
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
    if len(active_regions) == 0:
        active_regions = np.array(range(n_regions))
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


def reconfigure_model_with_fit_estimates(head, model_configuration, probabilistic_model, estimates,
                                         base_path, writer=None, plotter=None):
    # -------------------------- Reconfigure model after fitting:---------------------------------------------------
    for id_est, est in enumerate(ensure_list(estimates)):
        K = est.get("K", np.mean(model_configuration.K))
        tau1 = est.get("tau1", np.mean(model_configuration.tau1))
        tau0 = est.get("tau0", np.mean(model_configuration.tau0))
        # fit_conn = est.get("MC", model_configuration.connectivity)
        # if fit_conn.shape != model_configuration.connectivity.shape:
        #     temp_conn = model_configuration.connectivity
        #     temp_conn[probabilistic_model.active_regions][:, probabilistic_model.active_regions] = fit_conn
        #     fit_conn = temp_conn
        fit_conn = model_configuration.connectivity
        fit_model_configuration_builder = \
            ModelConfigurationBuilder(model_configuration.model_name, fit_conn,
                                      K_unscaled=K * model_configuration.number_of_regions). \
                set_parameter("tau1", tau1).set_parameter("tau0", tau0)
        x0 = model_configuration.x0
        x0[probabilistic_model.active_regions] = est["x0"].squeeze()
        x0_values_fit = fit_model_configuration_builder._compute_x0_values_from_x0_model(x0)
        hyp_fit = HypothesisBuilder().set_nr_of_regions(head.connectivity.number_of_regions). \
            set_name('fit' + str(id_est + 1)). \
            set_x0_hypothesis(list(probabilistic_model.active_regions),
                              x0_values_fit[probabilistic_model.active_regions]). \
            build_hypothesis()

        model_configuration_fit = \
            fit_model_configuration_builder.build_model_from_hypothesis(hyp_fit)

        if writer:
            writer.write_hypothesis(hyp_fit, path("fit_Hypothesis", base_path))
            writer.write_model_configuration(model_configuration_fit, path("fit_ModelConfig", base_path))

        # Plot nullclines and equilibria of model configuration
        if plotter:
            plotter.plot_state_space(model_configuration_fit, region_labels=head.connectivity.region_labels,
                                     special_idx=select_greater_values_array_inds(model_configuration_fit.x0),
                                     figure_name="fit_Nullclines and equilibria")  # threshold=X1EQ_CR_DEF),
        return model_configuration_fit


