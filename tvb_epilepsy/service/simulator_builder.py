import numpy
from tvb.datatypes import equations
from tvb.simulator import monitors, noise
from tvb.simulator.models import Epileptor

from tvb_epilepsy.base.constants.model_constants import model_noise_intensity_dict, VOIS, model_noise_type_dict
from tvb_epilepsy.base.constants.module_constants import NOISE_SEED, ADDITIVE_NOISE
from tvb_epilepsy.base.epileptor_models import EpileptorDPrealistic
from tvb_epilepsy.base.simulation_settings import SimulationSettings
from tvb_epilepsy.service.epileptor_model_factory import model_build_dict
from tvb_epilepsy.service.simulator.simulator_tvb import SimulatorTVB


class SimulatorBuilder(object):
    sim_type = ""
    model_name = "EpileptorDP"

    tau0 = 3000
    tau1 = 0.5
    time_length = 1500.0 / tau1

    fs = 10 * 2048.0 * tau1
    scale_fsavg = int(numpy.round(fs / 512.0))
    report_every_n_monitor_steps = 100.0

    zmode = "lin"
    pmode = "z"

    monitor_instance = monitors.TemporalAverage()
    monitor_expressions = VOIS[model_name]

    noise_instance = None
    noise_intensity = model_noise_intensity_dict[model_name]
    noise_type = model_noise_type_dict[model_name]

    def set_sim_type(self, sim_type):
        self.sim_type = sim_type
        return self

    def set_model_name(self, model_name):
        self.model_name = model_name
        return self

    def set_tau0(self, tau0):
        self.tau0 = tau0
        return self

    def set_tau1(self, tau1):
        self.tau1 = tau1
        return self

    def set_time_length(self, time_length):
        self.time_length = time_length
        return self

    def set_fs(self, fs):
        self.fs = fs
        return self

    def set_scale_fsavg(self, scale_fsavg):
        self.scale_fsavg = scale_fsavg
        return self

    def set_report_every_n_monitor_steps(self, report_every_n_monitor_step):
        self.report_every_n_monitor_steps = report_every_n_monitor_step
        return self

    def set_zmode(self, zmode):
        self.zmode = zmode
        return self

    def set_pmode(self, pmode):
        self.pmode = pmode
        return self

    def set_monitor_instance(self, monitor_instance):
        self.monitor_instance = monitor_instance
        return self

    def set_monitor_expressions(self, monitor_expressions):
        self.monitor_expressions = monitor_expressions
        return self

    def set_noise_instance(self, noise_instance):
        self.noise_instance = noise_instance
        return self

    def set_noise_intensiti(self, noise_intensity):
        self.noise_intensity = noise_intensity
        return self

    def set_noise_type(self, noise_type):
        self.noise_type = noise_type
        return self

    def _compute_time_scales(self):
        dt = 1000.0 / self.fs
        fsAVG = self.fs / self.scale_fsavg
        monitor_period = self.scale_fsavg * dt
        sim_length = self.time_length
        time_length_avg = numpy.round(sim_length / monitor_period)
        n_report_blocks = max(self.report_every_n_monitor_steps * numpy.round(time_length_avg / 100), 1.0)

        return dt, fsAVG, sim_length, monitor_period, n_report_blocks

    def build_simulator_tvb_from_model_configuration(self, model_configuration, connectivity):
        """
        Needs: connectivity, model_configuration, model, settings
        :return:
        """

        (dt, fsAVG, sim_length, monitor_period, n_report_blocks) = self._compute_time_scales()
        dt = 0.25 * dt

        model = model_build_dict[self.model_name](model_configuration, zmode=self.zmode)

        if isinstance(model, EpileptorDPrealistic):
            model.slope = 0.25
            model.pmode = self.pmode

        if self.sim_type == "realistic":
            if isinstance(model, Epileptor):
                model.tt = 0.2  # necessary to get spikes in a realistic frequency range
                model.r = 0.000025  # realistic seizures require a larger time scale separation
            else:
                # TODO: this is done again at the end of the method
                model.tau1 = 0.2
                model.tau0 = 40000.0

        if self.monitor_expressions is not None:
            monitor_expressions = [me.replace('lfp', 'x2 - x1') for me in self.monitor_expressions]
            model.variables_of_interest = monitor_expressions

        if monitor_period is not None:
            self.monitor_instance.period = monitor_period

        if model._ui_name == "EpileptorDP2D":
            if self.sim_type == "fast":
                self.noise_intensity *= 10
            elif self.sim_type == "fitting":
                self.noise_intensity = [0.0, 10 ** -3]

        if self.noise_instance is not None:
            self.noise_instance.nsig = self.noise_intensity
        else:
            if self.noise_type is ADDITIVE_NOISE:
                self.noise_instance = noise.Additive(nsig=self.noise_intensity,
                                                     random_stream=numpy.random.RandomState(seed=NOISE_SEED))
                self.noise_instance.configure_white(dt=dt)
            else:
                eq = equations.Linear(parameters={"a": 1.0, "b": 0.0})
                self.noise_instance = noise.Multiplicative(ntau=10, nsig=self.noise_intensity, b=eq,
                                                           random_stream=numpy.random.RandomState(seed=NOISE_SEED))
                noise_shape = self.noise_instance.nsig.shape
                self.noise_instance.configure_coloured(dt=dt, shape=noise_shape)

        settings = SimulationSettings(simulated_period=sim_length, integration_step=dt,
                                      noise_preconfig=self.noise_instance, noise_type=self.noise_type,
                                      noise_intensity=self.noise_intensity, noise_ntau=self.noise_instance.ntau,
                                      monitors_preconfig=self.monitor_instance,
                                      monitor_type=self.monitor_instance._ui_name,
                                      monitor_sampling_period=monitor_period,
                                      monitor_expressions=self.monitor_expressions,
                                      variables_names=model.variables_of_interest)

        simulator_instance = SimulatorTVB(connectivity, model_configuration, model, settings)

        simulator_instance.model.tau1 = self.tau1
        simulator_instance.model.tau0 = self.tau0

        simulator_instance.config_simulation(initial_conditions=None)

        return simulator_instance

    def build_simulator_java_from_model_configuration(self, model_configuration, connectivity):
        pass
