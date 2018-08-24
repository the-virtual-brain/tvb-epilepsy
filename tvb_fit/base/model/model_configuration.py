
import importlib

import numpy

from tvb_fit.base.utils.data_structures_utils import formal_repr


class ModelConfiguration(object):

    model_name = None
    connectivity = None
    coupling = None
    initial_conditions = None  # initial conditions in a reduced form
    noise = None
    monitor = None

    def __init__(self, model_name="Epileptor", connectivity=None, coupling=None,
                 monitor=None, initial_conditions=None, noise=None):
        self.model_name = model_name
        self.connectivity = connectivity
        self.coupling = coupling
        self.monitor = monitor
        self.initial_conditions = initial_conditions
        self.noise = noise

    @property
    def number_of_regions(self):
        if isinstance(self.connectivity, numpy.ndarray):
            return self.connectivity.shape[0]
        else:
            return 1

    def __repr__(self):
        d = {
            "01. model": self.model,
            "02. number_of_regions": self.number_of_regions,
            "03. connectivity": self.connectivity,
            "04. coupling": self.coupling,
            "05. monitor": self.monitor,
            "06. initial_conditions": self.initial_conditions,
            "07. noise": self.noise,
        }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def model(self):
        model_module = importlib.import_module('tvb.simulator.models.%s' % self.model_name.lower())
        model = getattr(model_module, self.model_name)
        return vars(model)

    def nvar(self):
        return self.model["nvar"]

    def set_parameter(self, pname, pval):
        setattr(self, pname, pval * numpy.ones((self.number_of_regions,)))

    def set_params_from_tvb_model(self, model, params):
        for pname in params:
            self.set_parameter(pname, getattr(model, pname))

    def set_attribute(self, attr_name, data):
        setattr(self, attr_name, data)
        return self
