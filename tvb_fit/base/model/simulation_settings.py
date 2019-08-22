import numpy

from tvb_fit.base.constants import WHITE_NOISE, NOISE_SEED

from tvb_scripts.utils.data_structures_utils import formal_repr


class SimulationSettings(object):
    """
    This class defines a convention of settings needed in order to create a generic Simulator (TVB or Java).
    """
    integrator_type = "Heun"

    def __init__(self, integrator_type="HeunStochastic", integration_step=0.01220703125, simulation_length=1000,
                 noise_ntau=0.0, noise_type=WHITE_NOISE, noise_seed=NOISE_SEED, noise_intensity=10 ** -6,
                 monitor_sampling_period=0.9765625, monitor_type="TemporalAverage", monitor_vois=numpy.array([])):
        self.integrator_type = integrator_type
        self.integration_step = integration_step
        self.simulation_length = simulation_length
        self.noise_type = noise_type
        self.noise_ntau = noise_ntau
        self.noise_intensity = noise_intensity
        self.noise_seed = noise_seed
        self.monitor_type = monitor_type
        self.monitor_sampling_period = monitor_sampling_period
        self.monitor_vois = monitor_vois

    def __repr__(self):
        d = {"01. integrator_type": self.integrator_type,
             "02. integration_step": self.integration_step,
             "03. simulation_length": self.simulation_length,
             "04. integrator_type": self.integrator_type,
             "05. noise_type": self.noise_type,
             "06. noise_ntau": self.noise_ntau,
             "07. noise_seed": self.noise_seed,
             "08. noise_intensity": self.noise_intensity,
             "09. monitor_type": self.monitor_type,
             "10. monitor_sampling_period": self.monitor_sampling_period,
             "11. monitor_vois": self.monitor_vois,
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def set_attribute(self, attr_name, data):
        setattr(self, attr_name, data)

    def monitor_expressions(self, model_vois):
        return (numpy.array(model_vois)[self.monitor_vois]).tolist()
