"""
@version $Id: simulators.py 1588 2016-08-18 23:44:14Z denis $

Mechanism for launching and configuring generic Simulations (it will have TVB or Episense implementations)
"""

from abc import ABCMeta, abstractmethod
from vep.base.model_vep import formal_repr
from collections import OrderedDict
from vep.base.constants import NOISE_SEED


class SimulationSettings(object):
# 
    def __init__(self, integration_step=0.01220703125, length=5000, 
                 monitor_sampling_period=0.9765625, monitor_expr=["y3 - y0", "y2"],
                 noise_preconfig=None, noise_seed=NOISE_SEED, noise_intensity=0.0001):
        self.integration_step = integration_step
        self.simulated_period = length
        self.monitor_sampling_period = monitor_sampling_period
        self.monitor_expr = monitor_expr
        self.integration_noise_seed = noise_seed
        self.noise_intensity = noise_intensity
        self.noise_preconfig = noise_preconfig
        
        

    def __repr__(self):
        d =  {"a. integration_step": self.integration_step,
              "b. monitor_sampling_period": self.monitor_sampling_period,
              "c. monitor_expr": self.monitor_expr,
              "f. integration_noise_seed": self.integration_noise_seed,
              "e. noise_intensity": self.noise_intensity,
              "d. simulated_period": self.simulated_period }
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0]) ) )

    def __str__(self):
        return self.__repr__()


class ABCSimulator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def launch_simulation(self, hypothesis, head, vep_settings=SimulationSettings()):
        pass

    @abstractmethod
    def launch_pse(self, hypothesis, head, vep_settings=SimulationSettings()):
        pass
