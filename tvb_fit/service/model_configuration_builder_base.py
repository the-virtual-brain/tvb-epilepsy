# coding=utf-8
"""

"""
from abc import ABCMeta, abstractmethod

from tvb_fit.base.utils.log_error_utils import initialize_logger
from tvb_fit.base.utils.data_structures_utils import formal_repr


class ModelConfigurationBuilderBase(object):
    __metaclass__ = ABCMeta

    logger = initialize_logger(__name__)

    def __init__(self, number_of_regions=1):
        self.number_of_regions = number_of_regions

    def __repr__(self):
        d = {"01. Number of regions": self.number_of_regions,
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    @abstractmethod
    def build_model_config_from_simulator(self, simulator):
      pass

    def set_attribute(self, attr_name, data):
        setattr(self, attr_name, data)

    # TODO: This is used from PSE for varying an attribute's value. We should find a better way, not hardcoded strings.
    def set_attributes_from_pse(self, values, paths, indices):
        for i, val in enumerate(paths):
            vals = val.split(".")
            if vals[0] == "model_configuration_builder":
                getattr(self, vals[1])[indices[i]] = values[i]
