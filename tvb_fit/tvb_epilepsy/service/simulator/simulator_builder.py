from copy import deepcopy
import numpy
from tvb.simulator import noise
from tvb.simulator.models.base import Model
from tvb.simulator.monitors import Monitor, TemporalAverage
from tvb.simulator.noise import Noise
from tvb_fit.base.constants import WHITE_NOISE, COLORED_NOISE, NOISE_SEED
from tvb_fit.base.utils.log_error_utils import initialize_logger, raise_value_error, warning
from tvb_fit.base.utils.data_structures_utils import isequal_string, ensure_list
from tvb_fit.base.model.simulation_settings import SimulationSettings
from tvb_fit.tvb_epilepsy.base.constants.model_constants import PMODE_DEF
from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation import compute_initial_conditions_from_eq_point
from tvb_fit.tvb_epilepsy.service.simulator.epileptor_model_factory \
    import model_noise_intensity_dict, VOIS, EPILEPTOR_MODEL_NVARS
from tvb_fit.tvb_epilepsy.service.simulator.simulator_java import SimulatorJava
from tvb_fit.tvb_epilepsy.service.simulator.simulator_tvb import SimulatorTVB


class SimulatorBuilder(object):
    logger = initialize_logger(__name__)

    def __init__(self, model_configuration, simulator="tvb"):
        self.model_config = deepcopy(model_configuration)
        self.simulator = simulator
        self.simulation_length = 2500
        self.fs = 16384.0
        self.fs_monitor = 1024.0

    @property
    def model_name(self):
        return self.model_config.model_name

    def set_model(self, model=None):
        if isinstance(model, Model):
            self.model_config.model_name = model._ui_name
            self.model_config = self.model_config.set_params_from_tvb_model(model)
        else:
            self.model_config.model_name = model
        self.model_config = self.model_config.update_initial_conditions()
        return self

    def set_simulation_length(self, simulation_length):
        self.simulation_length = simulation_length
        return self

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

    def _check_noise_intesity_size(self, noise_intensity):
        nn = len(ensure_list(noise_intensity))
        if nn != 1 and nn != EPILEPTOR_MODEL_NVARS[self.model_name]:
            raise_value_error(
                "Noise intensity is neither of size 1 nor of size equal to the number of model variables, "
                "\n but of size: " + str(nn) + "!")

    def generate_white_noise(self, noise_intensity):
        self._check_noise_intesity_size(noise_intensity)
        noise_instance = noise.Additive(nsig=noise_intensity, random_stream=numpy.random.RandomState(seed=NOISE_SEED))
        noise_instance.configure_white(dt=1.0 / self.fs)
        return noise_instance

    def generate_colored_noise(self, noise_intensity, ntau, **kwargs):
        self._check_noise_intesity_size(noise_intensity)
        noise_instance = noise.Additive(ntau=ntau, nsig=noise_intensity,
                                        random_stream=numpy.random.RandomState(seed=NOISE_SEED))
        noise_shape = noise_instance.nsig.shape
        noise_instance.configure_coloured(dt=1.0 / self.fs, shape=noise_shape)
        return noise_instance

    def build_sim_settings(self):
        dt, monitor_period = self.set_time_scales()
        return SimulationSettings(simulation_length=self.simulation_length, integration_step=dt,
                                  noise_type=WHITE_NOISE, noise_ntau=0.0, noise_seed=NOISE_SEED,
                                  noise_intensity=model_noise_intensity_dict[self.model_name],
                                  monitor_sampling_period=monitor_period)

    def set_noise(self, sim_settings, **kwargs):
        # Check if the user provides a preconfigured noise instance to override
        noise = kwargs.get("noise", None)
        if isinstance(noise, Noise):
            self._check_noise_intesity_size(noise.nsig)
            sim_settings.noise_intensity = noise.nsig
            if noise.ntau == 0:
                sim_settings.noise_type = WHITE_NOISE
            else:
                sim_settings.noise_type = COLORED_NOISE
            sim_settings.noise_ntau = noise.ntau
        else:
            if isequal_string(sim_settings.noise_type, COLORED_NOISE):
                noise = self.generate_colored_noise(sim_settings.noise_intensity,
                                                    sim_settings.noise_ntau, **kwargs)
            else:
                noise = self.generate_white_noise(sim_settings.noise_intensity)
            sim_settings.noise_ntau = noise.ntau
        return noise, sim_settings

    def generate_temporal_average_monitor(self, sim_settings):
        monitor = TemporalAverage()
        monitor.period = sim_settings.monitor_sampling_period
        monitor_vois = numpy.array(sim_settings.monitor_vois)
        n_model_vois = len(VOIS[self.model_name])
        monitor_vois = monitor_vois[monitor_vois<n_model_vois]
        if len(monitor_vois) == 0:
            monitor_vois = numpy.array(range(n_model_vois))
        monitor.variables_of_interest = numpy.array(monitor_vois)
        sim_settings.monitor_vois = numpy.array(monitor.variables_of_interest)
        return (monitor, ), sim_settings

    def set_tvb_monitor(self, sim_settings, monitor):
        monitor = (monitor,)
        sim_settings.monitor_sampling_period = monitor.period
        monitor_vois = numpy.union1d(monitor.variables_of_interest, sim_settings.monitor_vois)
        n_model_vois = len(VOIS[self.model_name])
        monitor_vois = monitor_vois[monitor_vois < n_model_vois]
        if len(monitor_vois) == 0:
            monitor.variables_of_interest = numpy.array(range(n_model_vois))
        else:
            monitor.variables_of_interest = monitor_vois
        sim_settings.monitor_vois = numpy.array(monitor.variables_of_interest)
        return monitor, sim_settings

    def set_monitor(self, sim_settings, monitor=None):
        # Check if the user provides a preconfigured set of monitor instances to override
        if isinstance(monitor, Monitor):
            return self.set_tvb_monitor(monitor, sim_settings)
        elif isinstance(monitor, tuple) or isinstance(monitor, list):
            return self.set_tvb_monitor(monitor[0], sim_settings)
        else:
            return self.generate_temporal_average_monitor(sim_settings)

    def build_simulator_TVB_from_model_sim_settings(self, connectivity, sim_settings,
                                                    **kwargs):
        monitors, sim_settings = self.set_monitor(sim_settings, kwargs.get("monitors", None))

        noise, sim_settings = self.set_noise(sim_settings, **kwargs)

        simulator_instance = SimulatorTVB(self.model_config, connectivity, sim_settings)
        simulator_instance.config_simulation(noise, monitors)

        return simulator_instance, sim_settings

    def build_simulator_TVB(self, connectivity, **kwargs):

        sim_settings = self.build_sim_settings()

        return self.build_simulator_TVB_from_model_sim_settings(connectivity, sim_settings, **kwargs)

    def build_simulator_java_from_model_configuration(self, connectivity, **kwargs):

        sim_settings = self.build_sim_settings()
        # sim_settings.noise_intensity = kwargs.get("noise_intensity", 1e-6)
        sim_settings.noise_intensity = kwargs.get("noise_intensity", numpy.array([0., 0., 5e-6, 0.0, 5e-6, 0.]))

        simulator_instance = SimulatorJava(connectivity, self.model_config, sim_settings)

        return simulator_instance, sim_settings

    def build_simulator(self, connectivity, **kwargs):
        if isequal_string(self.simulator, "java"):
            return self.build_simulator_java_from_model_configuration(connectivity, **kwargs)
        else:
            return self.build_simulator_TVB(connectivity, **kwargs)


def build_simulator_java(model_configuration, connectivity, **kwargs):
    return SimulatorBuilder(model_configuration, "java").set_model("JavaEpileptor"). \
                                        build_simulator_java_from_model_configuration(connectivity, **kwargs)


def build_simulator_TVB_default(model_configuration, connectivity, **kwargs):
    return SimulatorBuilder(model_configuration).build_simulator(connectivity, **kwargs)


def build_simulator_TVB_paper(model_configuration, connectivity, **kwargs):
    return SimulatorBuilder(model_configuration).set_model("Epileptor").build_simulator(connectivity, **kwargs)


def build_simulator_TVB_reduced(model_configuration, connectivity, **kwargs):
    sim_builder = \
        SimulatorBuilder(model_configuration).set_model("EpileptorDP2D").set_fs(4096.0).set_simulation_length(1000.0)
    sim_settings = sim_builder.build_sim_settings()
    return sim_builder.build_simulator_TVB_from_model_sim_settings(connectivity, sim_settings, **kwargs)


def build_simulator_TVB_fitting(model_configuration, connectivity, **kwargs):
    sim_builder = SimulatorBuilder(model_configuration).set_model("EpileptorDP2D").\
                                                set_fs(2048.0).set_fs_monitor(2048.0).set_simulation_length(300.0)
    sim_builder.model_config.tau0 = 30.0
    sim_builder.model_config.tau1 = 0.5
    sim_settings = sim_builder.build_sim_settings()
    sim_settings.noise_intensity = numpy.array([0.0, 1e-5])
    return sim_builder.build_simulator_TVB_from_model_sim_settings(connectivity, sim_settings, **kwargs)


def build_simulator_TVB_realistic(model_configuration, connectivity, **kwargs):
    sim_builder = SimulatorBuilder(model_configuration).set_model("EpileptorDPrealistic"). \
                                                                        set_fs(2048.0).set_simulation_length(60000.0)
    sim_builder.model_config.tau0 = 60000.0
    sim_builder.model_config.tau1 = 0.2
    sim_builder.model_config.slope = 0.25
    sim_builder.model_config.pmode = numpy.array(kwargs.pop("pmode", numpy.array[PMODE_DEF]))
    sim_settings = sim_builder.build_sim_settings()
    sim_settings.noise_type = COLORED_NOISE
    sim_settings.noise_ntau = 20
    # Necessary a more stable integrator:
    sim_settings.integrator_type = kwargs.pop("integrator", "Dop853Stochastic")
    return sim_builder.build_simulator_TVB_from_model_sim_settings(connectivity, sim_settings, **kwargs)

