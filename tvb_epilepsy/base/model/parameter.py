from tvb_epilepsy.base.constants.module_constants import MAX_SINGLE_VALUE, MIN_SINGLE_VALUE
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict

class Parameter(object):

    def __init__(self, name="Parameter", low=MIN_SINGLE_VALUE, high=MAX_SINGLE_VALUE, p_shape=(), **kwargs):
        if isinstance(name, basestring):
            self.name = name
        else:
            raise_value_error("Parameter type " + str(name) + " is not a string!")
        if low < high:
            self.low = low
            self.high = high
        else:
            raise_value_error("low (" + str(low) + ") is not smaller than high(" + str(high) + ")!")
        if isinstance(p_shape, tuple):
            self.p_shape = p_shape
        else:
            raise_value_error("Parameter's " + str(self.name) + " p_shape="
                              + str(p_shape) + " is not a shape tuple!")

    def __repr__(self):
        d = {"1. name": self.name,
             "2. low": self.low,
             "3. high": self.high,
             "4. shape": self.p_shape}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()
