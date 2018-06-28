import numpy
from tvb_fit.tvb_epilepsy.base.constants.model_constants import WHITE_NOISE, NOISE_SEED
from tvb_fit.base.utils.data_structures_utils import formal_repr


class SimulationSettings(object):
    """
    This class defines a convention of settings needed in order to create a generic Simulator (TVB or Java).
    """
    integrator_type = "Heun"

    def __init__(self, integration_step=0.01220703125, simulated_period=1000,
                 noise_ntau=0.0, noise_type=WHITE_NOISE, noise_seed=NOISE_SEED, noise_intensity=10 ** -6,
                 monitor_sampling_period=0.9765625, monitor_type="TemporalAverage", monitor_expressions=["x1", "z"],
                 initial_conditions=numpy.array([])):
        self.integration_step = integration_step
        self.simulated_period = simulated_period
        # self.integrator_type = integrator_type
        self.noise_type = noise_type
        self.noise_ntau = noise_ntau
        self.noise_intensity = noise_intensity
        self.noise_seed = noise_seed
        self.monitor_type = monitor_type
        self.monitor_sampling_period = monitor_sampling_period
        self.monitor_expressions = monitor_expressions
        self.initial_conditions = initial_conditions

    def __repr__(self):
        d = {"01. integration_step": self.integration_step,
             "02. simulated_period": self.simulated_period,
             "03. integrator_type": self.integrator_type,
             "04. noise_type": self.noise_type,
             "05. noise_ntau": self.noise_ntau,
             "06. noise_seed": self.noise_seed,
             "07. noise_intensity": self.noise_intensity,
             "08. monitor_type": self.monitor_type,
             "09. monitor_sampling_period": self.monitor_sampling_period,
             "10. monitor_expressions": self.monitor_expressions,
             "11. initial_conditions": self.initial_conditions,
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def set_attribute(self, attr_name, data):
        setattr(self, attr_name, data)
