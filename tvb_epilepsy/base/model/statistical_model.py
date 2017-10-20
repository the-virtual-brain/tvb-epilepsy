import numpy as np

from tvb_epilepsy.base.constants import STATISTICAL_MODEL_TYPES, OBSERVATION_MODEL_EXPRESSIONS, OBSERVATION_MODELS

from tvb_epilepsy.base.utils import raise_value_error, warning, formal_repr, sort_dict, ensure_list
from tvb_epilepsy.base.h5_model import convert_to_h5_model


class Parameter(object):

    def __init__(self, name, low=None, high=None, loc=None, scale=None, shape=(1,), distribution="uniform"):

        # TODO: better controls for the inputs given!

        if isinstance(name, basestring):
            self.name = name
        else:
            raise_value_error("Parameter name " + str(name) + " is not a string!")

        if isinstance(shape, tuple):
            self.shape = shape
        else:
            raise_value_error("Parameter's " + str(self.name) + " shape="
                              + str(shape) + " is not a shape tuple!")

        if isinstance(distribution, basestring):
            self.distribution = distribution
        else:
            raise_value_error("Parameter's " + str(self.name) + " distribution="
                              + str(distribution) + " is not a string!")

        if low is None:
            warning("Lowest value for parameter + " + self.name + " is -inf!")
            self.low = -np.inf
        else:
            self.low = low

        if high is None:
            warning("Highest value for parameter + " + self.name + " is inf!")
            self.high = np.inf
        else:
            self.high = high

        if np.all(self.low >= self.high):
            raise_value_error("Lowest value low=" + str(self.low) + " of  parameter " + self.name +
                              "is not smaller than the highest one high=" + str(self.high) + "!")

        low_not_inf = np.all(np.abs(self.low) < np.inf)
        high_not_inf = np.all(np.abs(self.high) < np.inf)

        if low_not_inf and high_not_inf:
            half = (self.low + self.high) / 2.0
        elif not(low_not_inf) and not(high_not_inf):
            half = 0.0
        elif not(low_not_inf):
            half = -1.0
        else:
            half = 1.0

        if loc is None:
            self.loc = half
            warning("Location of parameter + " + self.name + " is set as location=" + str(self.loc) + "!")
        else:
            if loc < self.low and loc > self.high:
                self.loc = loc
            else:
                raise_value_error("Parameter's " + str(self.name) + " location=" + str(loc)
                                  + "is not in the interval defined by the lowest and highest values "
                                  + str([self.low, self.high]) + "!")

        if scale is None:
            if self.loc == 0.0:
                if half == 0.0:
                    warning("Scale of parameter + " + self.name + " is set as scale=1.0!")
                    self.scale = 1.0
                else:
                    self.scale = np.abs(half)
                    warning("Scale of parameter + " + self.name + " is set as scale=" + str(self.scale) + "!")
            else:
                self.scale = np.abs(self.loc)
                warning("Scale of parameter + " + self.name + " is set as scale=abs(location)=" + str(self.scale) + "!")

        else:
            if self.scale >= 0.0:
                self.scale = scale
                if self.scale == 0.0:
                    warning("Scale of parameter + " + self.name + " is 0.0!")
            else:
                raise_value_error("Parameter's " + str(self.name) + " scale=" + str(scale) + "<0.0!")

    def __repr__(self):
        d = {"1. name": self.name,
             "2. low": self.low,
             "3. high": self.high,
             "4. location": self.loc,
             "5. scale": self.scale,
             "6. distribution": self.distribution,
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


class StatisticalModel(object):

    def __init__(self, name, type, parameters, n_regions=0, n_active_regions=0, n_signals=0, n_times=0,
                 euler_method="backward", observation_model="logpower", observation_expression="x1z_offset"):

        self.n_regions = n_regions
        self.n_active_regions = n_active_regions
        self.n_nonactive_regions = self.n_regions - self.n_active_regions
        self.n_signals = n_signals
        self.n_times = n_times

        if isinstance(name, basestring):
            self.name = name
        else:
            raise_value_error("Statistical model's name " + str(name) + " is not a string!")

        self.parameters = {}
        try:
            for p in ensure_list(parameters):
                if isinstance(p, Parameter):
                    self.parameters.update({p.name: p})
                else:
                    raise_value_error("Not valid Parameter object detected!")
        except:
            raise_value_error("Failed to set StatisticalModel parameters=\n" + str(parameters))
        self.n_parameters = len(self.parameters)

        if np.in1d(type, STATISTICAL_MODEL_TYPES):
            self.type = type
        else:
            raise_value_error("Statistical model's tupe " + str(type) + " is not one of the valid ones: "
                              + str(STATISTICAL_MODEL_TYPES) + "!")

        if np.in1d(euler_method, ["backward", "forward"]):
            self.euler_method = euler_method
        else:
            raise_value_error("Statistical model's euler_method " + str(euler_method) + " is not one of the valid ones: "
                              + str(["backward", "forward"]) + "!")

        if np.in1d(observation_expression, OBSERVATION_MODEL_EXPRESSIONS):
            self.observation_expression = observation_expression
        else:
            raise_value_error("Statistical model's observation expression " + str(observation_expression) +
                              " is not one of the valid ones: "
                              + str(OBSERVATION_MODEL_EXPRESSIONS) + "!")

        if np.in1d(observation_model, OBSERVATION_MODELS):
            self.observation_model = observation_model
        else:
            raise_value_error("Statistical model's observation expression " + str(observation_model) +
                              " is not one of the valid ones: "
                              + str(OBSERVATION_MODELS) + "!")

    def __repr__(self):
        d = {"1. name": self.name,
             "2. type": self.type,
             "3. number of regions": self.n_regions,
             "4. number of active regions": self.n_active_regions,
             "5. number of nonactive regions": self.n_nonactive_regions,
             "6. number of observation signals": self.n_signals,
             "7. number of time points": self.n_times,
             "8. euler_method": self.euler_method,
             "9. observation_expression": self.observation_expression,
             "10. observation_model": self.observation_model,
             "11. number of parameters": self.n_parameters,
             "12. parameters": [p.__str__ for p in self.parameters.items()]}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "StatisicalModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)
