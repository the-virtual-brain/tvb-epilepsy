
import os
from copy import deepcopy

import numpy as np

from tvb_fit.base.utils.log_error_utils import initialize_logger, warning
from tvb_fit.base.utils.data_structures_utils import isequal_string
from tvb_fit.base.utils.file_utils import wildcardit, move_overwrite_files_to_folder_with_wildcard
from tvb_fit.base.computations.analyzers_utils import interval_scaling
from tvb_fit.service.timeseries_service import TimeseriesService
from tvb_fit.service.simulator import ABCSimulator

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation import calc_eq_z, calc_x0
from tvb_fit.tvb_epilepsy.base.model.epileptor_models import EpileptorDP2D
from tvb_fit.tvb_epilepsy.base.model.timeseries import Timeseries
from tvb_fit.tvb_epilepsy.service.simulator.simulator_builder import build_simulator_TVB_realistic, \
    build_simulator_TVB_fitting, build_simulator_TVB_default, build_simulator_TVB_paper
from tvb_fit.tvb_epilepsy.service.workflow.workflow_configure_model import WorkflowConfigureModel


class WorkflowSimulate(WorkflowConfigureModel):

    def __init__(self, config=Config(), reader=None, writer=None, plotter=None):
        super(WorkflowSimulate, self).__init__(config, reader, writer, plotter)
        self._sim_folder = ""
        self._sim_title_prefix = ""
        self._sim_params = {"type": "paper", "rescale_x1eq": None, "seeg_gain_mode": "lin",
                            "hpf_flag": False, "hpf_low": 10.0, "hpf_high": 256.0,
                            "spectral_options": {"log_scale": True}}
        self._ts_filename = "SimTS"
        self._simulator = None
        self._sim_settings = None
        self._sim_ts = {}

    @property
    def sim_type(self):
        return self._sim_params.get("type", "paper")

    @property
    def sim_folder(self):
        if len(self._sim_folder) == 0:
            self._sim_folder = os.path.join(self.hypo_folder, "Simulation_" + self.sim_type)
        return self._sim_folder

    @property
    def sim_figsfolder(self):
        if self.sim_folder != self.hypo_folder:
            sim_foldername = self.sim_folder.split(os.sep)[-1]
        else:
            sim_foldername = ""
        sim_figsfolder = os.path.join(self.hypo_figsfolder, sim_foldername)
        return sim_figsfolder

    @property
    def sim_model_path(self):
        sim_model_filename = "SimModel"
        if isinstance(self._simulator, ABCSimulator):
            sim_model_filename = sim_model_filename.replace("SimModel", "SimModel_" + self._simulator.model._ui_name)
        return os.path.join(self.sim_folder, sim_model_filename + ".h5")

    @property
    def ts_filename(self):
        return self._ts_filename

    @property
    def ts_filepath(self):
        return os.path.join(self.sim_folder, self.ts_filename + ".h5")

    @property
    def ts_epi_filepath(self):
        return self.ts_filepath.replace("TS", "TSepi")

    def configure_simulator(self, write_sim_model=True):
        if self._write_flag(write_sim_model):
            self._ensure_folder(self.sim_folder)
            writer = self._writer
        else:
            writer = None
        self._simulator, self._sim_settings = \
            configure_simulator(self.modelconfig, self.connectivity, self._sim_params.get("rescale_x1eq", None),
                                self.sim_type, self._config, self._logger, writer, self.sim_model_path)

    @property
    def simulator(self):
        if not isinstance(self._simulator, ABCSimulator):
            self.configure_simulator()
        return self._simulator

    def _rescale_x1eq(self, model_config):
        return rescale_x1eq(model_config, self._sim_params["rescale_x1eq"])

    def simulate(self,  write_ts=True, write_ts_epi=True, plot_sim=True, plot_spectral_raster=False):
        if self._write_flag(write_ts or write_ts_epi):
            writer = self._writer
            if write_ts:
                self._ensure_folder(self._get_foldername(self.ts_filepath))
            if write_ts_epi:
                self._ensure_folder(self._get_foldername(self.ts_epi_filepath))
        else:
            writer = None
        if self._plot_flag(plot_sim):
            plotter = self._plotter
            self._ensure_folder(self.sim_figsfolder)
        else:
            plotter = None
        self._logger.info("\n\nSimulating...")
        return simulate(self.simulator, self.head, self._config, self._logger, self._reader,
                        writer, self.ts_filepath, self.ts_epi_filepath,
                        plotter, self.hypothesis.all_disease_indices, self.sim_figsfolder, self._sim_title_prefix,
                        plot_spectral_raster, **self._sim_params)

    @property
    def simulated_ts(self):
        if len(self._sim_ts) > 0:
            return self._sim_ts
        else:
            if self.ts_filepath is not None and os.path.isfile(self.ts_filepath):
                self._logger.info("\n\nLoading previously simulated time series from file: " + self.ts_filepath)
                sim_output = self._reader.read_timeseries(self.ts_filepath)
                seeg = TimeseriesService().compute_seeg(sim_output.get_source(), self.head.sensorsSEEG,
                                                        sum_mode=self._sim_params["seeg_gain_mode"])
                self._sim_ts = {"source": sim_output, "seeg": seeg}
            else:
                self._sim_ts = self.simulate()
            return self._sim_ts


def rescale_x1eq(model_config, rescale_x1eq):
    x1eq_min = np.min(model_config.x1eq)
    model_config.x1eq = \
        interval_scaling(model_config.x1eq, min_targ=x1eq_min, max_targ=rescale_x1eq,
                         min_orig=x1eq_min, max_orig=np.max(model_config.x1eq))
    zeq = calc_eq_z(model_config.x1eq, model_config.yc, model_config.Iext1, "2d",
                    np.zeros(model_config.zeq.shape), model_config.slope, model_config.a,
                    model_config.b, model_config.d)
    model_config.x0 = calc_x0(model_config.x1eq, zeq, model_config.K, model_config.connectivity,
                              model_config.zmode, z_pos=True, shape=zeq.shape, calc_mode="non_symbol")
    return model_config


def configure_simulator(input_modelconfig, connectivity, rescale_x1eq=None, sim_type="paper",
                        config=Config(), logger=None, writer=None, sim_model_path=""):
    if logger is None:
        initialize_logger(__name__, config.out.FOLDER_LOGS)
    # Choose model
    # Available models beyond the TVB Epileptor (they all encompass optional variations from the different papers):
    # EpileptorDP: similar to the TVB Epileptor + optional variations,
    # EpileptorDP2D: reduced 2D model, following Proix et all 2014 +optional variations,
    # EpleptorDPrealistic: starting from the TVB Epileptor + optional variations, but:
    #      -x0, Iext1, Iext2, slope and K become noisy state variables,
    #      -Iext2 and slope are coupled to z, g, or z*g in order for spikes to appear before seizure,
    #      -multiplicative correlated noise is also used
    # Optional variations:
    model_config = deepcopy(input_modelconfig)
    if rescale_x1eq:
        model_config = rescale_x1eq(model_config, rescale_x1eq)
    logger.info("\n\nConfiguring simulation...")
    if isequal_string(sim_type, "realistic"):
        simulator, sim_settings = build_simulator_TVB_realistic(model_config, connectivity)
    elif isequal_string(sim_type, "fitting"):
        simulator, sim_settings = build_simulator_TVB_fitting(model_config, connectivity)
    elif isequal_string(sim_type, "paper"):
        simulator, sim_settings = build_simulator_TVB_paper(model_config, connectivity)
    else:
        simulator, sim_settings = build_simulator_TVB_default(model_config, connectivity)
    if writer:
        if not os.path.isdir(os.path.dirname):
            sim_model_path = os.path.join(config.out.FOLDER_RES, "SimModel"+simulator.model._ui_name + ".h5")
        else:
            sim_model_path = sim_model_path.replace("Model", "Model"+simulator.model._ui_name)
        writer.write_simulator_model(simulator.model, sim_model_path, simulator.connectivity.number_of_regions)
    return simulator, sim_settings


def compute_and_write_seeg(source_timeseries, simulator, sensors_dict, writer=None, ts_epi_filepath="", **sim_params):
    ts_service = TimeseriesService()
    fsAVG = 1000.0 / source_timeseries.time_step
    if isinstance(simulator.model, EpileptorDP2D):
        sim_params["hpf_flag"] = False
    if sim_params.get("hpf_flag", False):
        sim_params["hpf_low"] = max(sim_params.get("hpf_low", 10.0),
                                     1000.0 / (source_timeseries.time_end - source_timeseries.time_start))
        sim_params["hpf_high"] = min(fsAVG / 2.0 - 10.0, sim_params.get("hpf_high"))

    seeg_data = \
        ts_service.compute_seeg(source_timeseries, sensors_dict, sum_mode=sim_params.get("seeg_gain_mode", "lin"))
    for s_name, seeg in seeg_data.items():
        if sim_params.get("hpf_flag", False):
            seeg_data[s_name] = ts_service.filter(seeg, sim_params["hpf_low"], sim_params["hpf_high"],
                                                  mode='bandpass', order=3)
        else:
            seeg_data[s_name] = seeg
        if writer and os.path.isfile(ts_epi_filepath):
            try:
                writer.write_ts_seeg_epi(seeg_data[s_name], source_timeseries.time_step,
                                         ts_epi_filepath, sensors_name=s_name)
            except:
                warning("Failed to write simulated SEEG timeseries to epi file %s!" % ts_epi_filepath)
    return seeg_data


def compute_seeg_and_write_ts_to_h5(timeseries, simulator, sensors_dict,
                                    writer=None, ts_filepath="", ts_epi_filepath=""):
    source_timeseries = timeseries.get_source()
    if writer:
        try:
            writer.write_ts(timeseries, timeseries.time_step, ts_filepath)
        except:
            warning("Failed to write simulated timeseries to file %s!" % ts_filepath)
        if os.path.isdir(os.path.dirname(ts_epi_filepath)):
            try:
                writer.write_ts_epi(timeseries, timeseries.time_step, ts_epi_filepath, source_timeseries)
            except:
                warning("Failed to write simulated timeseries to epi file %s!" % ts_epi_filepath)
    seeg_ts_all = compute_and_write_seeg(source_timeseries, simulator, sensors_dict, writer, ts_epi_filepath)
    return timeseries, seeg_ts_all


def simulate(simulator, head, config=Config(), logger=None,
             reader=None, writer=None, ts_filepath="", ts_epi_filepath="",
             plotter=None, all_disease_indices=[], sim_figsfolder="", title_prefix="", plot_spectral_raster=False,
             **sim_params):
    if logger is None:
        logger = initialize_logger(__name__, config.out.FOLDER_LOGS)
    seeg = []
    if reader and os.path.isfile(ts_filepath):
        logger.info("\n\nLoading previously simulated time series from file: " + ts_filepath)
        sim_output = reader.read_timeseries(ts_filepath)
        seeg = TimeseriesService().compute_seeg(sim_output.get_source(), head.sensorsSEEG,
                                                sum_mode=sim_params.get("seeg_gain_mode", "lin"))
    else:
        logger.info("\n\nSimulating %s..." % sim_params.get("type", "paper"))
        sim_output, status = simulator.launch_simulation(report_every_n_monitor_steps=100, timeseries=Timeseries)
        if not status:
            logger.warning("\nSimulation failed!")
        else:
            time = np.array(sim_output.time).astype("f")
            logger.info("\n\nSimulated signal return shape: %s", sim_output.shape)
            logger.info("Time: %s - %s", time[0], time[-1])
            sim_output, seeg = compute_seeg_and_write_ts_to_h5(sim_output, simulator, head.sensorsSEEG,
                                                               writer, ts_filepath, ts_epi_filepath)

    sim_ts = {"source": sim_output, "seeg": seeg}

    if plotter:
        # Plot results
        plotter.plot_simulated_timeseries(sim_output, simulator.model, all_disease_indices,
                                          seeg_dict=seeg, spectral_raster_plot=plot_spectral_raster,
                                          spectral_options=sim_params.get("spectral_options",
                                                                          {"log_scale": True}),
                                          title_prefix=title_prefix)
        if os.path.isdir(sim_figsfolder) and (sim_figsfolder != config.out.FOLDER_FIGURES):
            move_overwrite_files_to_folder_with_wildcard(sim_figsfolder,
                                                         os.path.join(config.out.FOLDER_FIGURES,
                                                                      wildcardit("Sim")))
    return sim_ts