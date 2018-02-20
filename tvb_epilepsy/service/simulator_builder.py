import numpy
from tvb.datatypes import equations
from tvb.simulator import noise
from tvb.simulator.monitors import Monitor, TemporalAverage
from tvb.simulator.noise import Noise, Additive, Multiplicative
from tvb_epilepsy.base.constants.model_constants import NOISE_SEED, WHITE_NOISE, COLORED_NOISE
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.simulation_settings import SimulationSettings
from tvb_epilepsy.service.epileptor_model_factory import model_build_dict, model_noise_intensity_dict, VOIS, \
                                                            AVAILABLE_DYNAMICAL_MODELS_NAMES, EPILEPTOR_MODEL_NVARS
from tvb_epilepsy.service.simulator.simulator_java import EpileptorModel, java_model_builder, SimulatorJava
from tvb_epilepsy.service.simulator.simulator_tvb import SimulatorTVB


class SimulatorBuilder(object):
    logger = initialize_logger(__name__)

    simulator = "tvb"
    model_name = "EpileptorDP"

    simulated_period = 1000
    fs = 8192.0
    fs_monitor = 512.0

    def __init__(self, simulator="tvb"):
        self.simulator = simulator

    def set_model_name(self, model_name):
        # TODO: check that model_name is one of the available ones
        if model_name not in AVAILABLE_DYNAMICAL_MODELS_NAMES:
            raise_value_error(model_name + " is not one of the available models: \n" +
                              str(AVAILABLE_DYNAMICAL_MODELS_NAMES) + " !")
        self.model_name = model_name
        return self

    def set_simulated_period(self, simulated_period):
        self.simulated_period = simulated_period
        return simulated_period

    def set_fs(self, fs):
        self.fs = fs
        return self

    def set_fs_monitor(self, fs_monitor):
        self.fs_monitor = fs_monitor
        return self

    def set_time_scales(self):
        scale_fsavg = int(numpy.round(self.fs / self.fs_monitor))
        dt = 1000.0 / self.fs
        monitor_period = scale_fsavg * dt
        return dt, monitor_period

    def generate_model(self, model_configuration, **kwargs):
        if isequal_string(self.model_name, EpileptorModel._ui_name) and not isequal_string(self.simulator, "java"):
            raise_value_error("Custom EpileptorModel can be used only with java simulator!")
        elif not isequal_string(self.model_name, EpileptorModel._ui_name) and isequal_string(self.simulator, "java"):
            raise_value_error("Only java EpileptorModel can be used with java simulator!")
        return model_build_dict[self.model_name](model_configuration, **kwargs)

    def generate_white_noise(self, noise_intensity):
        nn = len(noise_intensity)
        if nn != 1 and nn != EPILEPTOR_MODEL_NVARS[self.model_name]:
            raise_value_error("Noise intensity is neither of size 1 nor of size equal to the number of model variables, "
                              "\n but of size: " + str(nn) + "!")
        noise_instance = noise.Additive(nsig=noise_intensity, random_stream=numpy.random.RandomState(seed=NOISE_SEED))
        noise_instance.configure_white(dt=1.0/self.fs)
        return noise_instance

    def generate_colored_noise(self, noise_intensity, ntau, **kwargs):
        nn = len(noise_intensity)
        if nn != 1 and nn != EPILEPTOR_MODEL_NVARS[self.model_name]:
            raise_value_error(
                "Noise intensity is neither of size 1 nor of size equal to the number of model variables, "
                "\n but of size: " + str(nn) + "!")
        eq = equations.Linear(parameters=kwargs.get("parameters", {"a": 1.0, "b": 0.0}))
        noise_instance = noise.Multiplicative(ntau=ntau, nsig=noise_intensity, b=eq,
                                              random_stream=numpy.random.RandomState(seed=NOISE_SEED))
        noise_shape = noise_instance.nsig.shape
        noise_instance.configure_coloured(dt=1.0/self.fs, shape=noise_shape)
        return noise_instance

    def build_sim_settings(self):
        dt, monitor_period = self.set_time_scales()
        return SimulationSettings(simulated_period=self.simulated_period, integration_step=dt,
                                  noise_type=WHITE_NOISE, noise_ntau=0.0, noise_seed=NOISE_SEED,
                                  noise_intensity=model_noise_intensity_dict[self.model_name],
                                  monitor_sampling_period=monitor_period,
                                  monitor_expressions=VOIS[self.model_name])

    def set_noise(self, sim_settings, **kwargs):
        # Check if the user provides a preconfigured noise instance to override
        noise = kwargs.get("noise", None)
        if isinstance(noise, Noise):
            nn = len(noise.nsig)
            if nn != 1 and nn != EPILEPTOR_MODEL_NVARS[self.model_name]:
                raise_value_error(
                    "Noise intensity is neither of size 1 nor of size equal to the number of model variables, "
                    "\n but of size: " + str(nn) + "!")
            sim_settings.noise_intensity = noise.nsig
            if isinstance(noise, Additive):
                sim_settings.noise_type = WHITE_NOISE
            elif isinstance(noise, Multiplicative):
                sim_settings.noise_type = COLORED_NOISE
            sim_settings.noise_ntau = noise.ntau
        else:
            if isequal_string(sim_settings.noise_type , COLORED_NOISE):
                noise = self.generate_colored_noise(sim_settings.noise_intensity,
                                                    sim_settings.noise_ntau, **kwargs)
            else:
                noise = self.generate_white_noise(sim_settings.noise_intensity)
            sim_settings.noise_ntau = noise.ntau
        return noise, sim_settings

    def generate_temporal_average_monitor(self, monitor_sampling_period):
        monitor = TemporalAverage()
        monitor.period = monitor_sampling_period
        return monitor

    def set_monitor(self, model, sim_settings, monitors=None):
        model.variables_of_interest = [me.replace('lfp', 'x2 - x1') for me in sim_settings.monitor_expressions]
        # Check if the user provides a preconfigured set of monitor instances to override
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
            monitors = (self.generate_temporal_average_monitor(sim_settings.monitor_sampling_period), )
        return model, monitors, sim_settings

    def build_simulator_TVB_from_model_sim_settings(self, model_configuration, connectivity, model, sim_settings,
                                                    **kwargs):
        model, monitors, sim_settings = self.set_monitor(model, sim_settings, kwargs.get("monitors", None))

        noise, sim_settings = self.set_noise(sim_settings, **kwargs)

        simulator_instance = SimulatorTVB(connectivity, model_configuration, model, sim_settings)
        simulator_instance.config_simulation(noise, monitors, initial_conditions=None)

        return simulator_instance, sim_settings, model

    def build_simulator_TVB(self, model_configuration, connectivity, **kwargs):

        model = self.generate_model(model_configuration)

        sim_settings = self.build_sim_settings()

        return self.build_simulator_TVB_from_model_sim_settings(model_configuration, connectivity,
                                                                model, sim_settings, **kwargs)

    def build_simulator_java_from_model_configuration(self, model_configuration, connectivity):

        model = java_model_builder(model_configuration)

        noise_intensity = 1e-6  # numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.])

        sim_settings = self.build_sim_settings()

        simulator_instance = SimulatorJava(connectivity, model_configuration, model, sim_settings)

        return simulator_instance, sim_settings, model

    def build_simulator(self, model_configuration, connectivity, **kwargs):
        if isequal_string(self.simulator, "java"):
            return self.build_simulator_java_from_model_configuration(model_configuration, connectivity)
        else:
            return self.build_simulator_TVB(model_configuration, connectivity, **kwargs)



def build_simulator_TVB_default(model_configuration, connectivity, **kwargs):
    return SimulatorBuilder().build_simulator(model_configuration, connectivity, **kwargs)


def build_simulator_TVB_paper(model_configuration, connectivity, **kwargs):
    return SimulatorBuilder().set_model_name("Epileptor").build_simulator(model_configuration, connectivity, **kwargs)


def build_simulator_TVB_fitting(self, model_configuration, connectivity, **kwargs):
    sim_builder = SimulatorBuilder().set_model_name("EpileptorDP2D")
    model = sim_builder.generate_model(model_configuration, **kwargs)
    model.tau0 = 10.0
    model.tau1 = 0.5
    sim_settings = self.build_sim_settings()
    sim_settings.noise_intensity = 1e-3
    return sim_builder.build_simulator_TVB_from_model_sim_settings(model_configuration, connectivity,
                                                                   model, sim_settings, **kwargs)


def build_simulator_TVB_realistic(self, model_configuration, connectivity, **kwargs):
    sim_builder = \
        SimulatorBuilder().set_model_name("EpileptorDP2Drealistic").set_fs(4096.0).set_simulated_period(50000.0)
    model = sim_builder.generate_model(model_configuration, **kwargs)
    model.tau0 = 30000.0
    model.tau1 = 0.2
    model.slope = 0.25
    sim_settings = self.build_sim_settings()
    sim_settings.noise_type = COLORED_NOISE
    sim_settings.noise_ntau = 10
    return sim_builder.build_simulator_TVB_from_model_sim_settings(model_configuration, connectivity,
                                                                   model, sim_settings, **kwargs)

