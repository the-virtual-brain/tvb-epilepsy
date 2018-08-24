# coding=utf-8
"""

"""

import importlib

import numpy

from tvb_fit.base.utils.log_error_utils import initialize_logger, warning
from tvb_fit.base.utils.data_structures_utils import formal_repr
from tvb_fit.base.model.model_configuration import ModelConfiguration


class ModelConfigurationBuilder(object):

    logger = initialize_logger(__name__)

    model_name = None
    connectivity = None
    coupling = None
    initial_conditions = None  # initial conditions in a reduced form
    noise = None
    monitor = None

    def __repr__(self):
        d = {"01. model": self.model,
             "02. Number of regions": self.number_of_regions,
             "03. connectivity": self.connectivity,
             "04. coupling": self.coupling,
             "05. monitor": self.monitor,
             "06. initial_conditions": self.initial_conditions,
             "07. noise": self.noise
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    @property
    def number_of_regions(self):
        if isinstance(self.connectivity, numpy.ndarray):
            return self.connectivity.shape[0]
        else:
            return 1

    def model(self):
        model_module = importlib.import_module('tvb.simulator.models.%s' % self.model_name.lower())
        model = getattr(model_module, self.model_name)
        model = vars(model)
        model["model_name"] = self.model_name
        for key in model.keys():
            if key in ["_ui_name", "ui_configurable_parameters", "variables_of_interest", "state_variable_range",
                       "state_variables", "_nvar", "cvar", ] \
               or callable(model[key]):
                del model[key]
        return model

    def nvar(self):
        return self.model["nvar"]

    def set_parameter(self, pname, pval):
        setattr(self, pname, pval * numpy.ones((self.number_of_regions,)))

    def set_params_from_tvb_model(self, model, params):
        for pname in params:
            self.set_parameter(pname, getattr(model, pname))

    def build_model_config_from_tvb(self):
        model = self.model
        del model["model_name"]
        model_config = ModelConfiguration(self.model_name, self.connectivity, self.coupling,
                                          self.monitor, self.initial_conditions, self.noise, **model)
        return model_config

    def build_model_config_from_model_config(self, model_config):
        if not isinstance(model_config, dict):
            model_config_dict = model_config.__dict__
        else:
            model_config_dict = model_config
        model_configuration = ModelConfiguration()
        for attr, value in model_configuration.__dict__.items():
            value = model_config_dict.get(attr, None)
            if value is None:
                warning(attr + " not found in the input model configuration dictionary!" +
                        "\nLeaving default " + attr + ": " + str(getattr(model_configuration, attr)))
            if value is not None:
                setattr(model_configuration, attr, value)
        return model_configuration

    def set_attribute(self, attr_name, data):
        setattr(self, attr_name, data)
        return self

