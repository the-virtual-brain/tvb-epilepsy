
import sys

import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, isequal_string
from tvb_epilepsy.base.computations.statistics_utils import mean_std_to_distribution_params
from tvb_epilepsy.base.h5_model import convert_to_h5_model


AVAILABLE_DISTRIBUTIONS = ["uniform", "normal", "gamma", "lognormal", "exponential", "beta", "binomial", "chisquare",
                           "poisson", "bernoulli"]

MAX_VALUE = sys.float_info.max
MIN_VALUE = sys.float_info.min

# TODO: Rules and checks for low, high, loc, and scale parameters to agree with each other and with the pdf...


class Parameter(object):

    def __init__(self, name, pdf_params={}, low=MIN_VALUE, high=MAX_VALUE, shape=(1,), pdf_name="uniform"):
        if isinstance(name, basestring):
            self.name = name
        else:
            raise_value_error("Parameter name " + str(name) + " is not a string!")
        if isinstance(shape, tuple):
            self.shape = shape
        else:
            raise_value_error("Parameter's " + str(self.name) + " shape="
                              + str(shape) + " is not a shape tuple!")
        if isinstance(pdf_name, basestring):
            self.pdf = pdf_name
        else:
            raise_value_error("Parameter's " + str(self.name) + " pdf="
                              + str(pdf_name) + " is not a string!")
        self.low = low
        self.high = high
        self.pdf_params = pdf_params

    def __repr__(self):
        d = {"1. name": self.name,
             "2. low": self.low,
             "3. high": self.high,
             "4. pdf_params": self.pdf_params,
             "5. pdf": self.pdf,
             "6. shape": self.shape}
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

    # def get_pdf_samples(self, n_samples=100):
    #     x = np.linspace()
    #     # Following: https://stackoverflow.com/questions/25141250/
    #     # how-to-truncate-a-numpy-scipy-exponential-distribution-in-an-efficient-way
    #     # TODO: to have distributions parameters valid for the truncated distributions instead for the original one
    #     # pystan might be needed for that...
    #     rnd_cdf = np.random.uniform(getattr(ss, self.pdf)(**kwargs).cdf(x=trunc_limits.get("low", -np.inf)),
    #                          getattr(ss, self.pdf)(**kwargs).cdf(x=trunc_limits.get("high", np.inf)),
    #                          size=size)
    #     return getattr(ss, distribution)(**kwargs).ppf(q=rnd_cdf)