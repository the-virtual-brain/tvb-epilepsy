import numpy
from tvb.datatypes import equations
from tvb.simulator import noise
from tvb.simulator.models import Epileptor
from tvb_epilepsy.base.constants.model_constants import model_noise_intensity_dict, VOIS, model_noise_type_dict
from tvb_epilepsy.base.constants.model_constants import NOISE_SEED, ADDITIVE_NOISE
from tvb_epilepsy.base.epileptor_models import EpileptorDPrealistic
from tvb_epilepsy.base.simulation_settings import SimulationSettings
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.service.epileptor_model_factory import model_build_dict
from tvb_epilepsy.service.simulator.simulator_java import EpileptorModel, java_model_builder, SimulatorJava
from tvb_epilepsy.service.simulator.simulator_tvb import SimulatorTVB


class SimulatorBuilder(object):
    logger = initialize_logger(__name__)

    sim_type = "realistic"
    model_name = "EpileptorDPrealistic"

    time_length = 10000.0
    fs = 2048.0
    scale_fsavg = int(numpy.round(fs / 512.0))
    report_every_n_monitor_steps = 100.0

    n_report_blocks = 0

    zmode = "lin"
    pmode = "z"

    def set_sim_type(self, sim_type):
        self.sim_type = sim_type
        return self

    def set_model_name(self, model_name):
        self.model_name = model_name
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

    def _compute_time_scales(self):
        dt = 1000.0 / self.fs
        fsAVG = self.fs / self.scale_fsavg
        monitor_period = self.scale_fsavg * dt
        sim_length = self.time_length
        time_length_avg = numpy.round(sim_length / monitor_period)
        self.n_report_blocks = max(self.report_every_n_monitor_steps * numpy.round(time_length_avg / 100), 1.0)

        return dt, fsAVG, sim_length, monitor_period

    def _build_simulator_TVB_settings_and_model(self, model_configuration, dt, sim_length, monitor_period):
        model = model_build_dict[self.model_name](model_configuration, zmode=numpy.array(self.zmode))

        if isinstance(model, EpileptorDPrealistic):
            model.slope = 0.25
            model.pmode = numpy.array(self.pmode)

        if self.sim_type == "realistic":
            if isinstance(model, Epileptor):
                model.tt = 0.2  # necessary to get spikes in a realistic frequency range
                model.r = 0.000025  # realistic seizures require a larger time scale separation
            else:
                model.tau1 = 0.2
                model.tau0 = 40000.0

        monitor_expressions = VOIS[model._ui_name]
        monitor_expressions = [me.replace('lfp', 'x2 - x1') for me in monitor_expressions]
        model.variables_of_interest = monitor_expressions

        noise_intensity = model_noise_intensity_dict[self.model_name]
        noise_type = model_noise_type_dict[self.model_name]

        if model._ui_name == "EpileptorDP2D":
            if self.sim_type == "fast":
                noise_intensity *= 10
            elif self.sim_type == "fitting":
                noise_intensity = [0.0, 10 ** -3]

        if noise_type is ADDITIVE_NOISE:
            noise_instance = noise.Additive(nsig=noise_intensity,
                                            random_stream=numpy.random.RandomState(seed=NOISE_SEED))
            noise_instance.configure_white(dt=dt)
        else:
            eq = equations.Linear(parameters={"a": 1.0, "b": 0.0})
            noise_instance = noise.Multiplicative(ntau=10, nsig=noise_intensity, b=eq,
                                                  random_stream=numpy.random.RandomState(seed=NOISE_SEED))
            noise_shape = noise_instance.nsig.shape
            noise_instance.configure_coloured(dt=dt, shape=noise_shape)

        settings = SimulationSettings(simulated_period=sim_length, integration_step=dt,
                                      noise_preconfig=noise_instance, noise_type=noise_type,
                                      noise_intensity=noise_intensity, noise_ntau=noise_instance.ntau,
                                      monitor_sampling_period=monitor_period,
                                      monitor_expressions=monitor_expressions,
                                      variables_names=model.variables_of_interest)
        return settings, model

    def build_simulator_tvb_from_model_configuration(self, model_configuration, connectivity):

        (dt, fsAVG, sim_length, monitor_period) = self._compute_time_scales()

        sim_settings, model = self._build_simulator_TVB_settings_and_model(model_configuration, dt, sim_length,
                                                                           monitor_period)

        simulator_instance = SimulatorTVB(connectivity, model_configuration, model, sim_settings)
        simulator_instance.config_simulation(initial_conditions=None)

        return simulator_instance

    def build_simulator_TVB_fitting(self, model_configuration, connectivity):

        tau0 = 10.0
        tau1 = 0.5

        (dt, fsAVG, sim_length, monitor_period) = self._compute_time_scales()
        dt = 0.25 * dt

        settings, model = self._build_simulator_TVB_settings_and_model(model_configuration, dt, sim_length,
                                                                       monitor_period)

        simulator_instance = SimulatorTVB(connectivity, model_configuration, model, settings)
        simulator_instance.model.tau0 = tau0
        simulator_instance.model.tau1 = tau1
        simulator_instance.config_simulation(initial_conditions=None)

        return simulator_instance

    def build_simulator_TVB_realistic(self, model_configuration, connectivity):

        tau1 = 0.2
        tau0 = 40000

        (dt, fsAVG, sim_length, monitor_period) = self._compute_time_scales()
        dt = 0.25 * dt

        settings, model = self._build_simulator_TVB_settings_and_model(model_configuration, dt, sim_length,
                                                                       monitor_period)

        simulator_instance = SimulatorTVB(connectivity, model_configuration, model, settings)
        simulator_instance.model.tau0 = tau0
        simulator_instance.model.tau1 = tau1
        simulator_instance.config_simulation(initial_conditions=None)

        return simulator_instance

    def build_simulator_java_from_model_configuration(self, model_configuration, connectivity):
        if self.model_name != EpileptorModel._ui_name:
            self.logger.info("You can use only " + EpileptorModel._ui_name + "for custom simulations!")

        model = java_model_builder(model_configuration)

        noise_intensity = 0  # numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])

        (dt, fsAVG, sim_length, monitor_period) = self._compute_time_scales()

        settings = SimulationSettings(simulated_period=sim_length, integration_step=dt,
                                      noise_intensity=noise_intensity,
                                      monitor_sampling_period=monitor_period)

        simulator_instance = SimulatorJava(connectivity, model_configuration, model, settings)

        return simulator_instance
