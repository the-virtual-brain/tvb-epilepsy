
import os
from copy import deepcopy

import numpy as np

from tvb_fit.base.constants import PriorsModes, Target_Data_Type

from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants \
    import XModes, OBSERVATION_MODELS, compute_upsample, compute_seizure_length, TAU1_DEF, TAU0_DEF
from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_fit.tvb_epilepsy.base.model.epileptor_probabilistic_models \
    import EpiProbabilisticModel, ODEEpiProbabilisticModel, SDEEpiProbabilisticModel
from tvb_fit.tvb_epilepsy.service.model_inversion_services import \
    ModelInversionService, ODEModelInversionService, SDEModelInversionService
from tvb_fit.tvb_epilepsy.service.probabilistic_models_builders import \
    ProbabilisticModelBuilder, ODEProbabilisticModelBuilder, SDEProbabilisticModelBuilder
from tvb_fit.tvb_epilepsy.service.workflow.workflow_lsa import WorkflowLSA
from tvb_fit.tvb_epilepsy.service.workflow.workflow_simulation import configure_simulator, simulate

from tvb_scripts.utils.data_structures_utils import isequal_string, find_labels_inds, ensure_list
from tvb_scripts.utils.file_utils import wildcardit, move_overwrite_files_to_folder_with_wildcard


class WorkflowInvertModel(WorkflowLSA):

    def __init__(self, config=Config()):
        super(WorkflowInvertModel, self).__init__(config)
        self._fit_folder = ""
        self._standard_source2D_ts_filename = "StandardSource2Dts"
        self._fit_target_data_filename = "FitTargetData"
        self._probabilistic_model_filename = "ProblstcModel"
        self._fit_model_data_filename = "FitModelData"
        self._fit_sensors_id = 0
        self._fit_model_type = "SDE"
        self._fit_model_params = {"tau1": TAU1_DEF, "tau0": TAU0_DEF}
        self._fit_standard_source2D_ts = None
        self._model_inversion_service = None
        self._model_inversion_settings = {"active_regions_exlude":
                                              find_labels_inds(self.head.connectivity.region_labels, ["unknown"])}
        self._fit_target_data = None
        self._fit_data_settings = {"label_strip_fun": None, "exclude_channels": [], "sim_times_on_off": [50.0, 100.0],
                                   "times_on": [1500.0], "time_length": 700.0, "sensor_labels_selection": []}
        # "seeg_auto_selection": None, "downsampling": 2,
        self._fit_signals = None
        self._fit_signals_path = ""
        self._seizure_files = []
        self._fit_x1prior_ts = None
        self._probabilistic_model = None
        self._probabilistic_model_settings = {"xmode": XModes.X1EQMODE.value,
                                              "priors_mode": PriorsModes.NONINFORMATIVE.value,
                                              "target_data_type": Target_Data_Type.SYNTHETIC.value
                                              "observation_model": OBSERVATION_MODELS.SEEG_LOGPOWER.value}
        self._fit_sampler = None
        self._fit_function = "fit"
        self._fit_sampler_settings = {"method": "sample", "n_chains_or_runs": 4, "output_samples": 100,
                                      "num_warmup": 100, "min_samples_per_run_or_chain": 100, "iters": 100000,
                                      "max_depth": 15, "delta": 0.95, "tol_rel_obj": 1e-6}
        self._fit_sampler_debug = 1
        self._fit_sampler_simulate = 0
        self._fit_estimates = {}
        self._fit_samples = []
        self._fit_summary = []
        self._fit_information_criteria = []
        self._fit_plot_title_prefix = ""

    @property
    def fit_target_data_type(self):
        if isinstance(self._probabilistic_model, probabilistic_model_class(self._fit_model_type)):
            return self._probabilistic_model.target_data_type
        else:
            return self._probabilistic_model_settings.get("target_data_type", Target_Data_Type.SYNTHETIC.value)

    @property
    def fit_folder(self):
        if len(self._fit_folder) == 0:
            self._fit_folder = os.path.join(self.hypo_folder, "ModelInversion")
        return self._fit_folder

    @property
    def fit_figsfolder(self):
        if self._fit_folder != self.hypo_folder:
            fit_foldername = self.fit_folder.split(os.sep)[-1]
        else:
            fit_foldername = ""
        fit_figsfolder = os.path.join(self.hypo_figsfolder, fit_foldername)
        return fit_figsfolder

    @property
    def fit_sensors(self):
        return self.head.get_sensors_by_index(self._fit_sensors_id)

    @property
    def standard_source2D_ts_path(self):
        return os.path.join(self.res_folder, "Simulation_fitting", self._standard_source2D_ts_filename + ".h5")

    @property
    def fit_target_data_filepath(self):
        return os.path.join(self.fit_folder, self._fit_target_data_filename + ".h5")

    @property
    def fit_signals_filepath(self):
        if not os.path.isfile(self._fit_signals_path):
            self._fit_signals_path = os.path.join(self.fit_folder, "FitSignals.h5")
        return self._fit_signals_path

    @property
    def fit_signals_figsfolder(self):
        return os.path.join(self.fit_figsfolder, "DataProcessing")

    @property
    def probabilistic_model_filepath(self):
        return os.path.join(self.fit_folder, self._probabilistic_model_filename + ".h5")

    @property
    def fit_model_data_filepath(self):
        return os.path.join(self.fit_folder, self._fit_model_data_filename + ".h5")

    def simulate_standard_source2D(self, write_ts=True, plot_sim=True):
        if write_ts:
            self._ensure_folder(self._get_foldername(self.standard_source2D_ts_path))
            writer = self._writer
        else:
            writer = None
        if self._plot_flag(plot_sim):
            plotter = self._plotter
            sim_figsfolder = os.path.join(self.hypo_figsfolder, "Simulation_fitting"),
            self._ensure_folder(sim_figsfolder)
        else:
            plotter = None
            sim_figsfolder = ""
        return simulate_standard_source2D(self.modelconfig, self.head, self._fit_data_settings["sim_times_on_off"],
                                          self._config, self._logger, self._reader, writer=writer,
                                          standard_source2D_ts_path=self.standard_source2D_ts_path,
                                          plotter=plotter, sim_figsfolder=sim_figsfolder,
                                          all_disease_indices=self.hypothesis.all_disease_indices)

    @property
    def fit_standard_source2D_ts(self):
        if not(isinstance(self._fit_standard_source2D_ts, Timeseries)):
            try:
                self._standard_source2D_ts = self.reader(self.standard_source2D_ts_path)
            except:
                self._fit_standard_source2D_ts = self.simulate_standard_source2D()
        return self._fit_standard_source2D_ts

    @property
    def model_inversion_service(self):
        if not isinstance(self._model_inversion_service, model_inversion_class(self._fit_model_type)):
            self._model_inversion_service = self.generate_model_inversion_service()
        return self._model_inversion_service

    def update_model_inversion_service(self):
        for param, val in self._model_inversion_settings.items():
            setattr(self._model_inversion_service, param, val)
        return self._model_inversion_service

    def generate_model_inversion_service(self):
        self._model_inversion_service = model_inversion_class(self._fit_model_type).__init__()
        self._model_inversion_service = self.update_model_inversion_service()
        return self._model_inversion_service

    def generate_probabilistic_model(self, **kwargs):
        self._probabilistic_model = \
            generate_probabilistic_model(self.modelconfig, self._fit_model_params, self._fit_model_type,
                                         self._probabilistic_model_settings, **kwargs)
        return self._probabilistic_model

    @property
    def probabilistic_model(self):
        if not isinstance(self._probabilistic_model, probabilistic_model_class(self._fit_model_type)):
            try:
                self._probabilistic_model = \
                    self._reader.read_probabilistic_model(self.probabilistic_model_filepath)
                return self._probabilistic_model
            except:
                return self.generate_probabilistic_model()

    def set_simulated_fit_signals(self, write_signals=True, plot_signals=True):
        if self._write_flag(write_signals):
            writer = self._writer
            self._ensure_folder(self._get_foldername(self.fit_signals_filepath))
        else:
            writer = None
        if self._plot_flag(plot_signals):
            plotter = self._plotter
            self._ensure_folder(self.fit_signals_figsfolder)
        return set_simulated_fit_signals(self.probabilistic_model, self.head, self.fit_signals_filepath,
                                         self._config, self._logger, writer,
                                         plotter, self.fit_signals_figsfolder, **self._fit_data_settings)

    def set_empirical_fit_signals(self, write_signals=True, plot_signals=True):
        if self._write_flag(write_signals):
            writer = self._writer
            self._ensure_folder(self._get_foldername(self.fit_signals_filepath))
        else:
            writer = None
        if self._plot_flag(plot_signals):
            plotter = self._plotter
            self._ensure_folder(self.fit_signals_figsfolder)
        return set_empirical_fit_signals(self.probabilistic_model, self.fit_signals_filepath,  self.seizures_files,
                                         self._config, self._logger, writer,
                                         plotter, self.fit_signals_figsfolder, **self._fit_data_settings)

    def set_fit_signals(self, write_signals=True, plot_signals=True):
        if self.probabilistic_model.target_data_type == Target_Data_Type.SYNTHETIC.value:
            self._fit_signals, self._probabilistic_model = self.set_simulated_fit_signals(write_signals, plot_signals)
        else:
            self._fit_signals, self._probabilistic_model = self.set_empirical_fit_signals(write_signals, plot_signals)
        return self._fit_signals, self._probabilistic_model

    @property
    def fit_signals(self):
        if isinstance(self._fit_signals, Timeseries):
            return self._fit_signals
        else:
            try:
                self._fit_signals = self._reader.read_timeseries(self.fit_signals_filepath)
            except:
                self._fit_signals, self._probabilistic_model = self.set_target_timeseries()
            return self._fit_signals

    def set_target_data(self, write_target_data=True, plot_target_data=True):
        self._fit_target_data, self._probabilistic_model = \
            self._model_inversion_service.set_target_data_and_time(self.fit_signals, self.probabilistic_model,
                                                                   head=self.head, sensors=self.fit_sensors)
        if self._plot_flag(plot_target_data):
            self._plotter.plot_raster({'Target Signals': self._fit_target_data.squeezed}, self._fit_target_data.time,
                                      time_units=self._fit_target_data.time_unit,
                                      title=self._fit_plot_title_prefix + 'Fit-Target Signals raster',
                                      offset=0.1, labels=self._fit_target_data.space_labels)
            self._plotter.plot_timeseries({'Target Signals': self._fit_target_data.squeezed},
                                          self._fit_target_data.time,
                                          time_units=self._fit_target_data.time_unit,
                                          title=self._fit_plot_title_prefix + 'Fit-Target Signals',
                                          labels=self._fit_target_data.space_labels)
            self._plotter.plot_active_regions_gain_matrix(self._fit_sensors, self.region_labels,
                                                          title=self._fit_plot_title_prefix +
                                                                "Active regions -> Target data projection",
                                                          show_x_labels=True, show_y_labels=True,
                                                          x_ticks=self._fit_sensors. \
                                                            get_sensors_inds_by_sensors_labels(
                                                              self._fit_target_data.space_labels),
                                                          y_ticks=self._probabilistic_model.active_regions)
            if self._fit_folder:
                move_overwrite_files_to_folder_with_wildcard(self.fit_figsfolder,
                                                             self._config.out.FOLDER_FIGURES + "/*Target*")

        if self._write_flag(write_target_data):
            self._writer.write_timeseries(self._fit_target_data, self.fit_target_data_filepath)

    def set_priors_parameters(self, write_probabilstic_model=True, plot_probabilistic_model=True):
        self._probabilistic_model.time_length = self._fit_target_data.time_length
        self._probabilistic_model.upsample = \
            compute_upsample( self._probabilistic_model.time_length /  self._probabilistic_model.number_of_seizures,
                             compute_seizure_length( self._probabilistic_model.tau0),  self._probabilistic_model.tau0)

        self._probabilistic_model.parameters =  self._probabilistic_model_builder( self._probabilistic_model). \
            generate_parameters(self._probabilistic_model_settings["sampled_params"],
                                self._probabilistic_model.parameters,  self._fit_target_data,
                                self._fit_standard_source2D_ts, self._fit_x1prior_ts)

        if self._plot_flag(plot_probabilistic_model):
            self._plotter.plot_probabilistic_model(self._probabilistic_model,
                                                   self._fit_plot_title_prefix + "Probabilistic Model")
            if self._fit_folder:
                move_overwrite_files_to_folder_with_wildcard(self.fit_figsfolder,
                                                             self._config.out.FOLDER_FIGURES + "/*Probabilistic*")

        if self._write_flag(write_probabilstic_model):
            self._writer. \
                write_probabilistic_model(self._probabilistic_model,
                                          self._probabilistic_model.model_config.number_of_regions,
                                          self.probabilistic_model_filepath)


def simulate_standard_source2D(modelconfig, head, sim_times_on_off, config=Config(), logger=None, reader=None,
                               writer=None, standard_source2D_ts_path="",
                               plotter=None, sim_figsfolder="", all_disease_indices=[]):
    sim_folder = os.path.dirname(standard_source2D_ts_path)
    sim_model_path = os.path.join(sim_folder, "SimModelEpileptorDP2D")
    simulator, sim_settings =\
        configure_simulator(modelconfig, head.connectivity, -1.2, "fitting", config, logger, writer, sim_model_path)
    # Only simulate, without writing to file or plotting and get desired interval
    simTS = simulate(simulator, head, config, logger, reader)["source"].\
        get_time_window_by_units(sim_times_on_off[0], sim_times_on_off[1])
    if writer:
        writer.write_ts(simTS, simTS.time_step, standard_source2D_ts_path)
    if plotter:
        # Plot results
        plotter.plot_simulated_timeseries(simTS, simulator.model, all_disease_indices, title_prefix="StandardSource2D")
        if os.path.isdir(sim_figsfolder) and (sim_figsfolder != config.out.FOLDER_FIGURES):
            move_overwrite_files_to_folder_with_wildcard(sim_figsfolder, os.path.join(config.out.FOLDER_FIGURES,
                                                                                      wildcardit("Sim")))
    return simTS


def model_inversion_class(fit_model_type):
    if isequal_string(fit_model_type, "SDE"):
        return SDEModelInversionService
    elif isequal_string(fit_model_type, "ODE"):
        return ODEModelInversionService
    else:
        return ModelInversionService


def probabilistic_model_class(fit_model_type):
    if isequal_string(fit_model_type, "SDE"):
        return SDEEpiProbabilisticModel
    elif isequal_string(fit_model_type, "ODE"):
        return ODEEpiProbabilisticModel
    else:
        return EpiProbabilisticModel


def probabilistic_model_builder_class(fit_model_type):
    if isequal_string(fit_model_type, "SDE"):
        return SDEProbabilisticModelBuilder
    elif isequal_string(fit_model_type, "ODE"):
        return ODEProbabilisticModelBuilder
    else:
        return ProbabilisticModelBuilder


def generate_probabilistic_model(input_modelconfig, fit_model_params={}, fit_model_type="SDE",
                                 probabilistic_model_settings={}, **kwargs):
    model_config = deepcopy(input_modelconfig)
    for param, val in fit_model_params.items():
        setattr(model_config, param, val)
    probabilistic_model_builder = probabilistic_model_builder_class(fit_model_type).__init__(model_config=model_config)
    probabilistic_model_builder = \
        probabilistic_model_builder.set_attributes(**probabilistic_model_settings)
    probabilistic_model = probabilistic_model_builder.generate_model(generate_parameters=False, **kwargs)
    return probabilistic_model


def get_preprocesing(probabilistic_model, sim_type="paper" ):
    log_flag = probabilistic_model.observation_model == OBSERVATION_MODELS.SEEG_LOGPOWER.value
    if probabilistic_model.target_data_type == Target_Data_Type.EMPIRICAL.value:
        preprocessing = ["hpf", "abs-envelope", "convolve", "decimate"]
        if log_flag:
            preprocessing += ["log"]
    else:
        preprocessing = []
        if sim_type == "paper":
            preprocessing += ["convolve"]
        if probabilistic_model.observation_model in OBSERVATION_MODELS.SEEG.value:
            preprocessing += ["mean_center"]
        preprocessing += ["decimate"]
    return preprocessing


def concatenate_seizures(signals):
    # Concatenate only the labels that exist in all signals:
    labels = signals[0].space_labels
    for signal in signals[1:]:
        labels = np.intersect1d(labels, signal.space_labels)
    signals = TimeseriesService().concatenate_in_time(signals, labels)
    return signals


def set_empirical_data(empirical_file, ts_file, head, sensors_lbls, sensor_id=0, seizure_length=SEIZURE_LENGTH,
                       times_on_off=[], time_units="ms", label_strip_fun=None, exclude_seeg=[],
                       preprocessing=TARGET_DATA_PREPROCESSING, low_hpf=LOW_HPF, high_hpf=HIGH_HPF, low_lpf=LOW_LPF,
                       high_lpf=HIGH_LPF, bipolar=BIPOLAR, win_len=WIN_LEN, plotter=None, title_prefix=""):
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
                                                        label_strip_fun, exclude_seeg, preprocessing,
                                                        low_hpf, high_hpf, low_lpf, high_lpf, bipolar, win_len,
                                                        plotter, title_prefix)
        H5Writer().write_timeseries(signals, ts_file)
    move_overwrite_files_to_folder_with_wildcard(os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              "fitData_EmpiricalSEEG"),
                                                 os.path.join(plotter.config.out.FOLDER_FIGURES,
                                                              title_prefix.replace(" ", "_")) + "*")
    return signals


def set_empirical_fit_signals(probabilistic_model, head, fit_signals_filepath,  seizures_files,
                              config=Config(), logger=None, writer=None,
                              plotter=None, fit_signals_figsfolder="", title_prefix="", **fit_data_settings):
    empirical_files = ensure_list(ensure_list(seizures_files))
    n_seizures = len(empirical_files)
    fit_data_settings["times_on"] = ensure_list(fit_data_settings["times_on"])
    if len(fit_data_settings["times_on"] ) == n_seizures:
        times_on_off = []
        for time_on in fit_data_settings["times_on"] :
            times_on_off.append([time_on, time_on + fit_data_settings["time_length"]])
    else:
        times_on_off = n_seizures * [[fit_data_settings["time_length"]]]
    signals = []
    ts_filename = ts_file.split(".h5")[0]
    for empirical_file, time_on_off in zip(empirical_files, times_on_off):
        seizure_name = os.path.basename(empirical_file).split(".")[0]
        signals.append(set_empirical_data(empirical_file, "_".join([ts_filename, seizure_name]) + ".h5",
                                          head, sensors_lbls, sensor_id, seizure_length, time_on_off, time_units,
                                          label_strip_fun, exclude_seeg, preprocessing,
                                          low_hpf, high_hpf, low_lpf, high_lpf, bipolar, win_len,
                                          plotter, title_prefix))
    if n_seizures > 1:
        signals = concatenate_seizures(signals)
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
    else:
        signals = signals[0]

    return signals, n_seizures



