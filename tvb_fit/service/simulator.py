from abc import ABCMeta, abstractmethod

import numpy



class ABCSimulator(object):
    __metaclass__ = ABCMeta

    model_configuration = None

    def __init__(self, model_configuration):
        self.model_configuration = model_configuration

    def get_vois(self):
        # TODO: Confirm the path monitor.expressions
        return [me.replace('x2 - x1', 'source') for me in self.model_configuration.monitor.exressions]

    ###
    # Prepare for tvb-epilepsy epileptor_models initial conditions given this model_config's initial conditions
    ###

    @abstractmethod
    def configure_initial_conditions(self, history_length=1):
       pass

    @abstractmethod
    def launch_simulation(self, **kwargs):
        pass

    # @abstractmethod
    # def launch_pse(self, hypothesis, head):
    #     pass




