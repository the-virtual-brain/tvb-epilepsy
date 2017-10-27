import numpy as np

from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict, isequal_string
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.service.sampling_service import mean_std_to_distribution_params
from tvb_epilepsy.base.h5_model import convert_to_h5_model


class Parameter(object):

    def __init__(self, name, low=None, high=None, loc=None, scale=None, shape=(1,), pdf="uniform"):
        if isinstance(name, basestring):
            self.name = name
        else:
            raise_value_error("Parameter name " + str(name) + " is not a string!")
        if isinstance(shape, tuple):
            self.shape = shape
        else:
            raise_value_error("Parameter's " + str(self.name) + " shape="
                              + str(shape) + " is not a shape tuple!")
        if isinstance(pdf, basestring):
            self.pdf = pdf
        else:
            raise_value_error("Parameter's " + str(self.name) + " pdf="
                              + str(pdf) + " is not a string!")
        self.low = low
        self.high = high
        self.loc = loc
        self.scale = scale
        if self.low is not None and self.high is not None:
            if np.any(self.low >= self.high):
                raise_value_error("Lowest value low=" + str(self.low) + " of  parameter " + self.name +
                                  "is not smaller than the highest one high=" + str(self.high) + "!")
        if self.loc is not None:
            if self.low is not None:
                if np.any(self.loc < self.low):
                    raise_value_error("Parameter's " + str(self.name) + " location=" + str(self.loc)
                                      + "is smaller than the lowest value " + str(self.low) + "!")
            if self.high is not None:
                if np.any(self.loc > self.high):
                    raise_value_error("Parameter's " + str(self.name) + " location=" + str(self.loc)
                                      + "is greater than the highest value " + str(self.high) + "!")
        if self.scale is not None:
            if self.scale < 0.0:
                raise_value_error("Parameter's " + str(self.name) + " scale=" + str(scale) + "<0.0!")

    def __repr__(self):
        d = {"1. name": self.name,
             "2. low": self.low,
             "3. high": self.high,
             "4. location": self.loc,
             "5. scale": self.scale,
             "6. pdf": self.pdf,
             "7. shape": self.shape}
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

    def get_pdf_params(self):
        if isequal_string(self.pdf, "uniform"):
            return (self.low, self.high)
        elif isequal_string(self.pdf, "normal"):
            return (self.loc, self.scale)
        else:
            return mean_std_to_distribution_params(self.loc, self.scale, output="dict", gamma_mode="alpha_beta")