import numpy
from tvb_epilepsy.base.constants.model_constants import NOISE_SEED
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr


class SimulationSettings(object):
    """
    This class defines a convention of settings needed in order to create a generic Simulator (TVB or Java).
    """

    def __init__(self, integration_step=0.01220703125, simulated_period=5000, scale_time=1, integrator_type="",
                 noise_preconfig=None, noise_ntau=0.0, noise_type="", noise_seed=NOISE_SEED,
                 noise_intensity=10 ** -6,
                 monitors_preconfig=None, monitor_sampling_period=0.9765625, monitor_expressions="",
                 monitor_type="", variables_names="", initial_conditions=numpy.array([])):
        self.integration_step = integration_step
        self.simulated_period = simulated_period
        self.scale_time = scale_time
        self.integrator_type = integrator_type
        self.noise_preconfig = noise_preconfig
        self.noise_type = noise_type
        self.noise_ntau = noise_ntau
        self.noise_intensity = noise_intensity
        self.noise_seed = noise_seed
        self.monitors_preconfig = monitors_preconfig
        self.monitor_type = monitor_type
        self.monitor_sampling_period = monitor_sampling_period
        self.monitor_expressions = monitor_expressions
        self.variables_names = variables_names
        self.initial_conditions = initial_conditions

    def __repr__(self):
        d = {"01. integration_step": self.integration_step,
             "02. simulated_period": self.simulated_period,
             "03. scale_time": self.scale_time,
             "04. integrator_type": self.integrator_type,
             "05. noise_preconfig": self.noise_preconfig,
             "06. noise_type": self.noise_type,
             "07. noise_ntau": self.noise_ntau,
             "08. noise_seed": self.noise_seed,
             "09. noise_intensity": self.noise_intensity,
             "10. monitors_preconfig": self.monitors_preconfig,
             "11. monitor_type": self.monitor_type,
             "12. monitor_sampling_period": self.monitor_sampling_period,
             "13. monitor_expressions": self.monitor_expressions,
             "14. variables_names": self.variables_names,
             "15. initial_conditions": self.initial_conditions,
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def set_attribute(self, attr_name, data):
        setattr(self, attr_name, data)
