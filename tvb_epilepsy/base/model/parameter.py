
import sys

import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution \
                                                                                          import ProbabilityDistribution
from tvb_epilepsy.base.model.statistical_models.probability_distributions.uniform_distribution \
                                                                                              import UniformDistribution


class Parameter(object):

    def __init__(self, name, low=-np.inf, high=np.inf, shape=(1,), probability_distribution=None):
        if isinstance(name, basestring):
            self.name = name
        else:
            raise_value_error("Parameter name " + str(name) + " is not a string!")
        if low < high:
            self.low = low
            self.high = high
        else:
            raise_value_error("low (" + str(low) + ") is not smaller than high(" + str(high) + ")!")
        if isinstance(shape, tuple):
            self.shape = shape
        else:
            raise_value_error("Parameter's " + str(self.name) + " shape="
                              + str(shape) + " is not a shape tuple!")
        if isinstance(probability_distribution, ProbabilityDistribution):
            self.probability_distribution = probability_distribution
        else:
            raise_value_error("Parameter's " + str(self.name) + " probability distribution ="
                              + str(probability_distribution.name) +
                              "\n is not an instance of ProbabilityDistribution class!")
        if isinstance(probability_distribution, UniformDistribution):
            if np.any(self.probability_distribution.a < self.low):
                raise_value_error("Parameter's " + str(self.name) + " uniform distribution's parameter a (" +
                                  str(self.probability_distribution.a) +
                                  "\n does not match low value (" + str(self.low) + ")!")
            if np.any(self.probability_distribution.b > self.high):
                raise_value_error("Parameter's " + str(self.name) + " uniform distribution's parameter b (" +
                                  str(self.probability_distribution.b) +
                                  "\n does not match high value (" + str(self.high) + ")!")

    def __repr__(self):
        d = {"1. name": self.name,
             "2. low": self.low,
             "3. high": self.high,
             "4. probability distribution": self.probability_distribution,
             "5. shape": self.shape}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "ParameterModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)
