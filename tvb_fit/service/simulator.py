from abc import ABCMeta, abstractmethod


class ABCSimulator(object):
    __metaclass__ = ABCMeta

    model_configuration = None
    connectivity = None
    settings = None

    def __init__(self, model_configuration, connectivity, settings):
        self.model_configuration = model_configuration
        self.connectivity = connectivity
        self.settings = settings

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




