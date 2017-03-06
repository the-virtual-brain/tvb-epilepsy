"""
Mechanism for launching and configuring generic Simulations (it will have TVB or custom implementations)
"""

from abc import ABCMeta, abstractmethod
from tvb_epilepsy.base.model_vep import formal_repr
from tvb_epilepsy.base.constants import NOISE_SEED
from collections import OrderedDict


class SimulationSettings(object):
# 
    def __init__(self, integration_step=0.01220703125, simulated_period=5000,
                 monitors_preconfig=None, monitor_sampling_period=0.9765625, monitor_expr=None,
                 noise_preconfig=None, integration_noise_seed=NOISE_SEED, noise_intensity=10 ** -6):
        self.integration_step = integration_step
        self.simulated_period = simulated_period
        self.monitor_sampling_period = monitor_sampling_period
        self.monitor_expr = monitor_expr
        self.integration_noise_seed = integration_noise_seed
        self.noise_intensity = noise_intensity
        self.monitors_preconfig = monitors_preconfig
        self.noise_preconfig=noise_preconfig

    def __repr__(self):
        d =  {"a. integration_step": self.integration_step,
              "b. simulated_period": self.simulated_period,
              "c. monitors_preconfig": self.monitors_preconfig,
              "d. monitor_sampling_period": self.monitor_sampling_period,
              "e. monitor_expr": self.monitor_expr,
              "f. noise_preconfig": self.noise_preconfig,
              "g. integration_noise_seed": self.integration_noise_seed,
              "h. noise_intensity": self.noise_intensity }
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0]) ) )

    def __str__(self):
        return self.__repr__()


class ABCSimulator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def launch_simulation(self, hypothesis, head):
        pass

    @abstractmethod
    def launch_pse(self, hypothesis, head):
        pass
