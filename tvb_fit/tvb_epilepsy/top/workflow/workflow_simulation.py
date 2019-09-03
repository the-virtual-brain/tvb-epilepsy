
import os

from tvb_fit.service.simulator import ABCSimulator

from tvb_fit.tvb_epilepsy.base.constants.config import Config
from tvb_fit.tvb_epilepsy.top.scripts.simulation_scripts import rescale_x1eq, configure_simulator, \
    simulate
from tvb_fit.tvb_epilepsy.top.workflow.workflow_configure_model import WorkflowConfigureModel

from tvb_scripts.model.virtual_head.sensors import SensorTypes
from tvb_scripts.service.timeseries_service import TimeseriesService


class WorkflowSimulate(WorkflowConfigureModel):

    def __init__(self, config=Config(), reader=None, writer=None, plotter=None):
        super(WorkflowSimulate, self).__init__(config, reader, writer, plotter)
        self._sim_folder = ""
        self._sim_title_prefix = ""
        self._sim_params = {"type": "paper", "x1eq_rescale": None, "seeg_gain_mode": "lin",
                            "hpf_flag": False, "hpf_low": 10.0, "hpf_high": 256.0,
                            "spectral_options": {"log_scale": True}}
        self._ts_filename = "SimTS"
        self._simulator = None
        self._sim_settings = None
        self._sim_ts = {}
        self._sim_compute_seeg = True

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
            configure_simulator(self.modelconfig, self.connectivity, self._sim_params.get("x1eq_rescale", None),
                                self.sim_type, self.sim_model_path, writer, self._logger, self._config)

    @property
    def simulator(self):
        if not isinstance(self._simulator, ABCSimulator):
            self.configure_simulator()
        return self._simulator

    def _rescale_x1eq(self, model_config):
        return rescale_x1eq(model_config, self._sim_params["x1eq_rescale"])

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
        return simulate(self.simulator, self.head, self.hypothesis.all_disease_indices,
                        self._sim_compute_seeg, plot_spectral_raster,
                        self.ts_filepath, self.ts_epi_filepath, self.sim_figsfolder, self._sim_title_prefix,
                        self._reader, writer, plotter, self._logger, self._config, **self._sim_params)

    @property
    def simulated_ts(self):
        if len(self._sim_ts) > 0:
            return self._sim_ts
        else:
            if self.ts_filepath is not None and os.path.isfile(self.ts_filepath):
                self._logger.info("\n\nLoading previously simulated time series from file: " + self.ts_filepath)
                sim_output = self._reader.read_timeseries(self.ts_filepath)
                seeg = TimeseriesService().compute_seeg(sim_output.get_source(),
                                                        self.head.get_sensors(s_type=SensorTypes.TYPE_SEEG.value),
                                                        sum_mode=self._sim_params["seeg_gain_mode"])
                self._sim_ts = {"source": sim_output, "seeg": seeg}
            else:
                self._sim_ts = self.simulate()
            return self._sim_ts
