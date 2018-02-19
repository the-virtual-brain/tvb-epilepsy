import numpy
from tvb.datatypes import equations
from tvb.simulator import noise
from tvb.simulator.noise import Noise, Additive, Multiplicative
from tvb_epilepsy.base.constants.model_constants import NOISE_SEED, ADDITIVE_NOISE, MULTIPLICATIVE_NOISE
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.simulation_settings import SimulationSettings
from tvb_epilepsy.service.epileptor_model_factory import model_build_dict
from tvb_epilepsy.service.simulator.simulator_java import EpileptorModel, java_model_builder, SimulatorJava
from tvb_epilepsy.service.simulator.simulator_tvb import SimulatorTVB


class SimulatorBuilder(object):
    logger = initialize_logger(__name__)

    sim_type = "default"
    time_length = 1000.0  # 10000.0
    fs = 4096.0

    def __init__(self, sim_type="default"):
        self.sim_type = sim_type

    def set_fs(self, fs):
        self.fs = fs
        return self

    def set_time_length(self, time_length):
        self.time_length = time_length
        return self

    def _compute_time_scales(self, **kwargs):
        self.time_length = kwargs.get("time_length", self.time_length)
        self.fs = kwargs.get("fs", self.fs)
        scale_fsavg = kwargs.get("scale_fsavg", int(numpy.round(self.fs / 512.0)))
        dt = 1000.0 / self.fs
        fsAVG = self.fs / scale_fsavg
        monitor_period = scale_fsavg * dt
        return dt, fsAVG, monitor_period

    def _set_white_noise(self, dt, noise_intensity):
        noise_instance = noise.Additive(nsig=noise_intensity, random_stream=numpy.random.RandomState(seed=NOISE_SEED))
        noise_instance.configure_white(dt=dt)
        return noise_instance, ADDITIVE_NOISE

    def _set_colored_noise(self, dt, noise_intensity, **kwargs):
        eq = equations.Linear(parameters=kwargs.get("parameters", {"a": 1.0, "b": 0.0}))
        noise_instance = noise.Multiplicative(ntau=kwargs.get("ntau", 10), nsig=noise_intensity, b=eq,
                                              random_stream=numpy.random.RandomState(seed=NOISE_SEED))
        noise_shape = noise_instance.nsig.shape
        noise_instance.configure_coloured(dt=dt, shape=noise_shape)
        return noise_instance, MULTIPLICATIVE_NOISE

    def _build_simulator_TVB_settings_and_model(self, model_configuration, connectivity, model,
                                                noise_instance, noise_type, monitor_period, dt, **kwargs):

        # Check if the user provides a preconfigured noise instance to override
        if isinstance(kwargs.get("noise"), Noise):
            noise_instance = noise
            if isinstance(kwargs.get("noise"), Additive):
                noise_type = ADDITIVE_NOISE
            elif isinstance(kwargs.get("noise"), Multiplicative):
                noise_type = MULTIPLICATIVE_NOISE

        sim_settings = SimulationSettings(simulated_period=self.time_length, integration_step=dt,
                                          noise_preconfig=noise_instance, noise_type=noise_type,
                                          noise_intensity=noise_instance.nsig, noise_ntau=noise_instance.ntau,
                                          monitor_sampling_period=monitor_period,
                                          monitor_expressions=model.variables_of_interest,
                                          variables_names=model.variables_of_interest)

        simulator_instance = SimulatorTVB(connectivity, model_configuration, model, sim_settings)
        simulator_instance.config_simulation(initial_conditions=None)

        return simulator_instance, sim_settings, model

    def build_simulator_tvb_default(self, model_configuration, connectivity, **kwargs):
        model = model_build_dict[kwargs.get("model_name", "EpileptorDP")](model_configuration, **kwargs)

        monitor_expressions = kwargs.get("monitor_expressions", ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp'])
        model.variables_of_interest = [me.replace('lfp', 'x2 - x1') for me in monitor_expressions]

        dt, fsAVG, monitor_period = self._compute_time_scales(**kwargs)

        noise_instance, noise_type = self._set_white_noise(dt, kwargs.get("noise_intensity",
                                                                          numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])))

        return self._build_simulator_TVB_settings_and_model(model_configuration, connectivity, model,
                                                            noise_instance, noise_type, monitor_period, dt, **kwargs)

    def build_simulator_tvb_paper(self, model_configuration, connectivity, **kwargs):
        model = model_build_dict[kwargs.get("model_name", "Epileptor")](model_configuration, **kwargs)

        monitor_expressions = kwargs.get("monitor_expressions", ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp'])
        model.variables_of_interest = [me.replace('lfp', 'x2 - x1') for me in monitor_expressions]

        dt, fsAVG, monitor_period = self._compute_time_scales(**kwargs)

        noise_instance, noise_type = self._set_white_noise(dt, kwargs.get("noise_intensity",
                                                                          numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])))

        return self._build_simulator_TVB_settings_and_model(model_configuration, connectivity, model,
                                                            noise_instance, noise_type, monitor_period, dt, **kwargs)

    def build_simulator_tvb_fitting(self, model_configuration, connectivity, **kwargs):
        model = model_build_dict[kwargs.get("model_name", "EpileptorDP2D")](model_configuration, **kwargs)
        if model._ui_name == "Epileptor":
            model.r = kwargs.get("r", 1.0 / kwargs.get("tau0", 10.0))
            model.tt = kwargs.get("tau1", kwargs.get("tt", 0.5))
        else:
            model.tau0 = kwargs.get("tau0", 10.0)
            model.tau1 = kwargs.get("tau1", 0.5)

        monitor_expressions = kwargs.get("monitor_expressions", ['x1', 'z'])
        model.variables_of_interest = [me.replace('lfp', 'x1') for me in monitor_expressions]

        dt, fsAVG, monitor_period = self._compute_time_scales(**kwargs)

        noise_instance, noise_type = self._set_white_noise(dt, kwargs.get("noise_intensity",
                                                                          numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])))

        return self._build_simulator_TVB_settings_and_model(model_configuration, connectivity, model,
                                                            noise_instance, noise_type, monitor_period, dt, **kwargs)

    def build_simulator_tvb_realistic(self, model_configuration, connectivity, **kwargs):
        model = model_build_dict[kwargs.get("model_name", "EpileptorDP2Drealistic")](model_configuration, **kwargs)
        if model._ui_name == "Epileptor":
            model.r = kwargs.get("r", 1.0 / kwargs.get("tau0", 40000.0))
            model.tt = kwargs.get("tau1", kwargs.get("tt", 0.2))
            monitor_expressions = kwargs.get("monitor_expressions", ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp'])
        else:
            model.tau0 = kwargs.get("tau0", 40000.0)
            model.tau1 = kwargs.get("tau1", 0.2)
            monitor_expressions = kwargs.get("monitor_expressions", ['x1', 'y1', 'z', 'x2', 'y2', 'g', 'lfp',
                                                                     'x0_t', 'slope_t', 'Iext1_t', 'Iext2_t', 'K_t'])
        model.slope = kwargs.get("slope", 0.25)

        model.variables_of_interest = [me.replace('lfp', 'x2 - x1') for me in monitor_expressions]

        self.set_fs(4096.0).set_time_length(60000)
        dt, fsAVG, monitor_period = self._compute_time_scales(**kwargs)

        noise_instance, noise_type = \
            self._set_colored_noise(dt, kwargs.get("noise_intensity",
                                              numpy.array([0., 0., 1e-8, 0.0, 1e-8, 0., 1e-9, 1e-4, 1e-9, 1e-4, 1e-9])),
                                    **kwargs)

        return self._build_simulator_TVB_settings_and_model(model_configuration, connectivity, model,
                                                            noise_instance, noise_type, monitor_period, dt, **kwargs)

    def build_preconfig_simulator_TVB(self, model_configuration, connectivity, *kwargs):

        # Configure model, noise, time scales and monitors
        if isequal_string(self.sim_type, "paper"):
            return self.build_simulator_tvb_paper(model_configuration, connectivity, **kwargs)
        elif isequal_string(self.sim_type, "fitting"):
            return self.build_simulator_tvb_fitting(model_configuration, connectivity, **kwargs)
        elif isequal_string(self.sim_type, "realistic"):
            return self.build_simulator_tvb_realistic(model_configuration, connectivity, **kwargs)
        else:
            return self.build_simulator_tvb_default(model_configuration, connectivity, **kwargs)

    def build_simulator_java_from_model_configuration(self, model_configuration, connectivity, **kwargs):
        if kwargs.get("model_name", EpileptorModel._ui_name) != EpileptorModel._ui_name:
            self.logger.info("You can use only " + EpileptorModel._ui_name + "for custom simulations!")

        model = java_model_builder(model_configuration)

        noise_intensity = 0  # numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])

        dt, fsAVG, monitor_period = self._compute_time_scales(**kwargs)

        settings = SimulationSettings(simulated_period=self.time_length, integration_step=dt,
                                      noise_intensity=noise_intensity,
                                      monitor_sampling_period=monitor_period)

        simulator_instance = SimulatorJava(connectivity, model_configuration, model, settings)

        return simulator_instance, settings, model
