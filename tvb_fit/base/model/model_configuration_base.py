from abc import ABCMeta  # , abstractmethod


class ModelConfigurationBase(object):
    __metaclass__ = ABCMeta

    model = None
    connectivity = None  # A tvb-fit virtual patient connectivity
    coupling = None
    monitor = None
    initial_conditions = None  # initial conditions in a reduced form
    noise = None

    def __init__(self, model, connectivity, coupling=None, monitor=None, initial_conditions=None, noise=None):
        self.model = model
        self.connectivity = connectivity
        self.coupling = coupling
        self.monitor = monitor
        self.initial_conditions = initial_conditions
        self.noise = noise

    def get_vois(self):
        # TODO: Confirm the path monitor.expressions
        return [me.replace('x2 - x1', 'source') for me in self.monitor.exressions]



