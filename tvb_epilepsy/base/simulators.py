"""
Mechanism for launching and configuring generic Simulations (it will have TVB or custom implementations)
"""
import numpy
from abc import ABCMeta, abstractmethod
from tvb_epilepsy.base.model_vep import formal_repr
from tvb_epilepsy.base.constants import NOISE_SEED
from collections import OrderedDict
from tvb_epilepsy.base.equilibrium_computation import calc_equilibrium_point

class SimulationSettings(object):
# 
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
        d =  {"01. integration_step": self.integration_step,
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

    ###
    # Prepare for tvb-epilepsy epileptor_models initial conditions
    ###

    def prepare_initial_conditions(self, hypothesis, history_length=1):
        # Set default initial conditions right on the resting equilibrium point of the model...
        # ...after computing the equilibrium point (and correct it for zeql for a >=6D model
        initial_conditions = calc_equilibrium_point(self.model, hypothesis)
        #-------------------The lines below are for a specific "realistic" demo simulation:---------------------------------
        #if isinstance(model,EpileptorDPrealistic):
        #   shape = initial_conditions[6].shape
        #   type = initial_conditions[6].dtype
        #   initial_conditions[6] = 0.0** numpy.ones(shape,dtype=type) # hypothesis.x0.T
        #   initial_conditions[7] = 1.0 * numpy.ones((1,hypothesis.n_regions))#model.slope * numpy.ones((hypothesis.n_regions,1))
        #   initial_conditions[9] = 0.0 * numpy.ones((1,hypothesis.n_regions))#model.Iext2.T * numpy.ones((hypothesis.n_regions,1))
        # ------------------------------------------------------------------------------------------------------------------
        initial_conditions = numpy.expand_dims(initial_conditions, 2)
        initial_conditions = numpy.tile(initial_conditions, (history_length, 1, 1, 1))
        return initial_conditions