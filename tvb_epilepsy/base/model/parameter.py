
import numpy as np

from tvb_epilepsy.base.constants import MAX_SINGLE_VALUE
from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, sort_dict
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution \
    import ProbabilityDistribution
from tvb_epilepsy.base.model.statistical_models.probability_distributions.uniform_distribution \
    import UniformDistribution


class Parameter(object):

    def __init__(self, name="Parameter", low=-MAX_SINGLE_VALUE, high=MAX_SINGLE_VALUE, shape=(1,),
                 probability_distribution=None):
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
        if isinstance(probability_distribution, (ProbabilityDistribution, basestring)):
            self.probability_distribution = probability_distribution
            if isinstance(probability_distribution, ProbabilityDistribution):
                try:
                    if (np.ones(self.shape) + np.ones(probability_distribution.pdf_shape)).shape != self.shape:
                        raise_value_error("Parameter's (" + str(self.shape) +
                                          ") and distribution's (" + str(probability_distribution.pdf_shape) +
                                          ") shapes do not match!")
                except:
                    raise_value_error("Parameter's (" + str(self.shape) +
                                      ") and distribution's (" + str(probability_distribution.pdf_shape) +
                                      ") shapes do not propagate!")
                if isinstance(probability_distribution, UniformDistribution):
                    if np.any(self.probability_distribution.low > self.low):
                        # TODO: Decide whether we should throw an error here instead of the warning
                        warning("Parameter's " + str(self.name) + " uniform distribution's parameter a (" +
                                str(self.probability_distribution.low) +
                                "\n does not match low value (" + str(self.low) + ")!" +
                                "\nAdjusting probability distribution accordingly!")
                        self.probability_distribution.update_params(low=self.low)
                    if np.any(self.probability_distribution.high < self.high):
                        warning("Parameter's " + str(self.name) + " uniform distribution's parameter b (" +
                                str(self.probability_distribution.high) +
                                "\n does not match high value (" + str(self.high) + ")!" +
                                "\nAdjusting probability distribution accordingly!")
                        self.probability_distribution.update_params(high=self.high)

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

    # def get_distrib_params(self):
    #     if len(ensure_list(self.probability_distribution))==1:
    #         return self.probability_distribution.params
    #     else:
    #         params = []
    #         for pd in self.probability_distribution.flatten().tolist():
    #             params.append(pd.params)
    #         return np.reshape(params, self.shape)
    #
    # def get_distrib_stats(self, stat):
    #     if len(ensure_list(self.probability_distribution))==1:
    #         return getattr(self.probability_distribution, stat)
    #     else:
    #         stats = []
    #         for pd in self.probability_distribution.flatten().tolist():
    #             stats.append(getattr(pd, stat))
    #         return np.reshape(stats, self.shape)
