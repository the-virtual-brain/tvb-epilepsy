import numpy
from tvb.datatypes import equations
from tvb.simulator import noise
from tvb.simulator.monitors import Monitor, TemporalAverage
from tvb.simulator.noise import Noise, Additive, Multiplicative
from tvb_epilepsy.base.constants.model_constants import NOISE_SEED, WHITE_NOISE, COLORED_NOISE, VOIS, \
                                                                                            model_noise_intensity_dict
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.simulation_settings import SimulationSettings
from tvb_epilepsy.service.epileptor_model_factory import model_build_dict
from tvb_epilepsy.service.simulator.simulator_java import EpileptorModel, java_model_builder, SimulatorJava
from tvb_epilepsy.service.simulator.simulator_tvb import SimulatorTVB


class SimulatorBuilder(object):
    logger = initialize_logger(__name__)

    simulator = "tvb"

    def __init__(self, simulator="tvb"):
        self.simulator = simulator

    def _set_time_scales(self, **kwargs):
        fs = kwargs.get("fs", 4096.0)
        scale_fsavg = kwargs.get("scale_fsavg", int(numpy.round(fs / 512.0)))
        dt = 1000.0 / fs
        fsAVG = fs / scale_fsavg
        monitor_period = scale_fsavg * dt
        return dt, fsAVG, monitor_period

    def _generate_model(self, model_name, model_configuration, simulator="TVB", **kwargs):
        if isequal_string(model_name, "EpileptorModel._ui_name ") and not(isequal_string(simulator, "Custom")):
            raise_value_error("Custom EpileptorModel can be used only with Custom simulator!")
        elif not(isequal_string(model_name, "EpileptorModel._ui_name ")) and isequal_string(simulator, "Custom"):
            raise_value_error("Only Custom EpileptorModel can be used with Custom simulator!")
        return model_build_dict[model_name](model_configuration, **kwargs)

    def _set_white_noise(self, dt, noise_intensity):
        noise_instance = noise.Additive(nsig=noise_intensity, random_stream=numpy.random.RandomState(seed=NOISE_SEED))
        noise_instance.configure_white(dt=dt)
        return noise_instance

    def _set_colored_noise(self, dt, noise_intensity, ntau, **kwargs):
        eq = equations.Linear(parameters=kwargs.get("parameters", {"a": 1.0, "b": 0.0}))
        noise_instance = noise.Multiplicative(ntau=ntau, nsig=noise_intensity, b=eq,
                                              random_stream=numpy.random.RandomState(seed=NOISE_SEED))
        noise_shape = noise_instance.nsig.shape
        noise_instance.configure_coloured(dt=dt, shape=noise_shape)
        return noise_instance

    def _set_noise(self, sim_settings, **kwargs):
        # Check if the user provides a preconfigured noise instance to override
        noise = kwargs.get("noise", None)
        if isinstance(noise, Noise):
            if isinstance(noise, Additive):
                sim_settings.noise_type = WHITE_NOISE
            elif isinstance(noise, Multiplicative):
                sim_settings.noise_type = COLORED_NOISE
            sim_settings.noise_ntau = noise.ntau
            sim_settings.noise_intensity = noise.nsig
        else:
            if isequal_string(sim_settings.noise_type , COLORED_NOISE):
                noise = self._set_colored_noise(sim_settings.integration_step, sim_settings.noise_intensity,
                                                sim_settings.noise_ntau, **kwargs)
            else:
                noise = self._set_white_noise(sim_settings.integration_step, sim_settings.noise_intensity)
            sim_settings.noise_ntau = noise.ntau
        return noise, sim_settings

    def _set_monitor(self, model, sim_settings, **kwargs):
        model.variables_of_interest = [me.replace('lfp', 'x2 - x1') for me in sim_settings.monitor_expressions]
        monitors = kwargs.get("monitor", None)
        if isinstance(monitors, Monitor):
            monitors = (monitors,)
            sim_settings.monitor_sampling_period = monitors.period
        elif isinstance(monitors, tuple) or isinstance(monitors, list):
            what_to_watch = []
            sim_settings.monitor_sampling_period = []
            for monitor in monitors:
                if isinstance(monitor, Monitor):
                    what_to_watch.append(monitor)
                    sim_settings.monitor_sampling_period.append(monitor.period)
                what_to_watch = tuple(what_to_watch)
            monitors = what_to_watch
        else:
            monitors = TemporalAverage()
            monitors.period = sim_settings.monitor_sampling_period
            monitors = (monitors,)
        return model, monitors, sim_settings

    def build_simulator_model_settings_tvb_default(self, model_configuration, **kwargs):
        model = self._generate_model(kwargs.get("model_name", "EpileptorDP"), model_configuration, **kwargs)
        monitor_expressions = kwargs.get("monitor_expressions",VOIS[model._ui_name])
        dt, fsAVG, monitor_period = self._set_time_scales(**kwargs)
        noise_intensity = kwargs.get("noise_intensity", model_noise_intensity_dict[model._ui_name])
        sim_settings = SimulationSettings(simulated_period=kwargs.get("time_length", 1000), integration_step=dt,
                                          noise_type=kwargs.get("noise_type", WHITE_NOISE), noise_ntau=0.0,
                                          noise_intensity=noise_intensity,
                                          monitor_sampling_period=monitor_period,
                                          monitor_expressions=monitor_expressions)
        return model, sim_settings

    def build_simulator_model_settings_tvb_paper(self, model_configuration, **kwargs):
        model = self._generate_model(kwargs.get("model_name", "Epileptor"), model_configuration, **kwargs)
        monitor_expressions = kwargs.get("monitor_expressions",VOIS[model._ui_name])
        dt, fsAVG, monitor_period = self._set_time_scales(**kwargs)
        noise_intensity = kwargs.get("noise_intensity", model_noise_intensity_dict[model._ui_name])
        sim_settings = SimulationSettings(simulated_period=kwargs.get("time_length", 1000), integration_step=dt,
                                          noise_type=kwargs.get("noise_type", WHITE_NOISE), noise_ntau=0.0,
                                          noise_intensity=noise_intensity,
                                          monitor_sampling_period=monitor_period,
                                          monitor_expressions=monitor_expressions)
        return model, sim_settings

    def build_simulator_model_settings_tvb_fitting(self, model_configuration, **kwargs):
        model = self._generate_model(kwargs.get("model_name", "EpileptorDP2D"), model_configuration, **kwargs)
        if model._ui_name == "Epileptor":
            model.r = kwargs.get("r", 1.0 / kwargs.get("tau0", 10.0))
            model.tt = kwargs.get("tau1", kwargs.get("tt", 0.5))
        else:
            model.tau0 = kwargs.get("tau0", 10.0)
            model.tau1 = kwargs.get("tau1", 0.5)
        monitor_expressions = kwargs.get("monitor_expressions", VOIS[model._ui_name])
        dt, fsAVG, monitor_period = self._set_time_scales(**kwargs)
        noise_intensity = kwargs.get("noise_intensity", model_noise_intensity_dict[model._ui_name])
        sim_settings = SimulationSettings(simulated_period=kwargs.get("time_length", 1000), integration_step=dt,
                                          noise_type=kwargs.get("noise_type", WHITE_NOISE), noise_ntau=0.0,
                                          noise_intensity=noise_intensity,
                                          monitor_sampling_period=monitor_period,
                                          monitor_expressions=monitor_expressions)
        return model, sim_settings

    def build_simulator_model_settings_tvb_realistic(self, model_configuration, **kwargs):
        model = self._generate_model(kwargs.get("model_name", "EpileptorDPrealistic"), model_configuration, **kwargs)
        if model._ui_name == "Epileptor":
            model.r = kwargs.get("r", 1.0 / kwargs.get("tau0", 30000.0))
            model.tt = kwargs.get("tau1", kwargs.get("tt", 0.2))
        elif model._ui_name == "EpileptorDP2D":
            raise_value_error("Realistic simulation are not possible with the 2D reduction model EpileptorDP2D!")
        else:
            model.tau0 = kwargs.get("tau0", 30000.0)
            model.tau1 = kwargs.get("tau1", 0.2)
        model.slope = kwargs.get("slope", 0.25)
        monitor_expressions = kwargs.get("monitor_expressions",VOIS[model._ui_name])
        dt, fsAVG, monitor_period = self._set_time_scales(**kwargs)
        noise_intensity = kwargs.get("noise_intensity", model_noise_intensity_dict[model._ui_name])
        sim_settings = SimulationSettings(simulated_period=kwargs.get("time_length", 60000), integration_step=dt,
                                          noise_type=kwargs.get("noise_type", COLORED_NOISE),
                                          noise_ntau=kwargs.get("noise_ntau", 10.0),
                                          noise_intensity=noise_intensity,
                                          monitor_sampling_period=monitor_period,
                                          monitor_expressions=monitor_expressions)
        return model, sim_settings

    def build_preconfig_simulator_settings_for_TVB(self, model_configuration, **kwargs):
        sim_type = kwargs.get("sim_type", "default")
        # Configure model, noise, time scales and monitors
        if isequal_string(sim_type, "paper"):
            return self.build_simulator_model_settings_tvb_paper(model_configuration, **kwargs)
        elif isequal_string(sim_type, "fitting"):
            return self.build_simulator_model_settings_tvb_fitting(model_configuration, **kwargs)
        elif isequal_string(sim_type, "realistic"):
            return self.build_simulator_model_settings_tvb_realistic(model_configuration, **kwargs)
        else:
            return self.build_simulator_model_settings_tvb_default(model_configuration, **kwargs)

    def build_simulator_TVB_from_model_sim_settings(self, model_configuration, connectivity, model, sim_settings,
                                                    **kwargs):
        model, monitors, sim_settings = self._set_monitor(model, sim_settings, **kwargs)

        noise, sim_settings = self._set_noise(sim_settings, **kwargs)

        simulator_instance = SimulatorTVB(connectivity, model_configuration, model, sim_settings)
        simulator_instance.config_simulation(noise, monitors, initial_conditions=None)

        return simulator_instance, sim_settings, model

    def build_simulator_TVB(self, model_configuration, connectivity, **kwargs):

        model = kwargs.get("model", None)
        sim_settings = kwargs.get("sim_settings", None)

        model_pre_config, sim_settings_pre_config \
            = self.build_preconfig_simulator_settings_for_TVB(model_configuration, **kwargs)

        if sim_settings is None:
            sim_settings = sim_settings_pre_config

        if model is None:
            model = model_pre_config

        return self.build_simulator_TVB_from_model_sim_settings(model_configuration, connectivity,
                                                                model, sim_settings, **kwargs)

    def build_simulator_java_from_model_configuration(self, model_configuration, connectivity, **kwargs):
        if kwargs.get("model_name", EpileptorModel._ui_name) != EpileptorModel._ui_name:
            self.logger.info("You can use only " + EpileptorModel._ui_name + "for custom simulations!")

        model = java_model_builder(model_configuration)

        noise_intensity = 1e-6  # numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])

        dt, fsAVG, monitor_period = self._set_time_scales(**kwargs)

        settings = SimulationSettings(simulated_period=kwargs.get("time_length", 1000), integration_step=dt,
                                      noise_intensity=noise_intensity,
                                      monitor_sampling_period=monitor_period)

        simulator_instance = SimulatorJava(connectivity, model_configuration, model, settings)

        return simulator_instance, settings, model

    def build_simulator(self, model_configuration, connectivity, **kwargs):
        if isequal_string(self.simulator, "java"):
            return self.build_simulator_java_from_model_configuration(model_configuration, connectivity, **kwargs)
        else:
            return self.build_simulator_TVB(model_configuration, connectivity, **kwargs)
