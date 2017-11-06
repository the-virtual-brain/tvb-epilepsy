
from collections import OrderedDict

import numpy as np
import scipy as scp
import scipy.stats as ss

from tvb_epilepsy.base.constants import MAX_SYSTEM_VALUE
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.utils.data_structures_utils import dict_str, formal_repr, shape_to_size
from tvb_epilepsy.base.h5_model import convert_to_h5_model


logger = initialize_logger(__name__)


class SamplingService(object):

    def __init__(self, n_samples=10):
        self.sampler = None
        self.sampling_module = ""
        self.n_samples = n_samples
        self.shape = (1, n_samples)
        self.stats = {}

    def __repr__(self):
        d = {"01. Sampling module": self.sampling_module,
             "02. Sampler": self.sampler,
             "03. Number of samples": self.n_samples,
             "04. Samples' shape": self.shape,
             }
        return formal_repr(self, d) + "\n05. Resulting statistics: " + dict_str(self.stats)

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model({"sampling_module": self.sampling_module, "sampler": self.sampler,
                                        "n_samples": self.n_samples, "shape": self.shape, "stats": self.stats})
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def adjust_shape(self, parameter_shape):
        shape = []
        for p in parameter_shape:
            shape.append(p)
        shape.append(self.n_samples)
        self.shape = shape

    def check_for_infinite_bounds(self, low, high):
        low = np.array(low)
        high = np.array(high)
        id = (low == -np.inf)
        if np.any(id):
            warning("Sampling is not possible with infinite bounds! Setting lowest system value for low!")
            low[id] = -MAX_SYSTEM_VALUE
        id = (high == np.inf)
        if np.any(id):
            warning(
                "Sampling is not possible with infinite bounds! Setting highest system value for high!")
            high[id] = MAX_SYSTEM_VALUE
        return low, high

    def check_size(self, low, high, parameter_shape):
        n_params = shape_to_size((low + high).shape)
        shape_size = shape_to_size(parameter_shape)
        if shape_size > n_params:
            warning("Input parameters' size (" + str(n_params) +
                    ") larger than the one implied by input parameters's shape (" + str(shape_size) + ")!" +
                    "\nModifying input parameters' size accordingly!")
            n_params = shape_size
        elif n_params > shape_size:
            warning("Input parameters' size (" + str(n_params) +
                    ") smaller than the one implied by input parameters's shape (" + str(shape_size) + ")!" +
                    "\nModifying input parameters' shape accordingly!: " + str((n_params,)))
            parameter_shape = (n_params,)
        i1 = np.ones(parameter_shape)
        low = low * i1
        high = high * i1
        return low, high, n_params, parameter_shape

    def compute_stats(self, samples):
        return OrderedDict([("mean", samples.mean(axis=-1)), ("median", scp.median(samples, axis=-1)),
                            ("std", samples.std(axis=-1)), ("var", samples.var(axis=-1)),
                            ("kurt", ss.kurtosis(samples, axis=-1)), ("skew", ss.skew(samples, axis=-1)),
                            ("min", samples.min(axis=-1)), ("max", samples.max(axis=-1)),
                            ("1%", np.percentile(samples, 1, axis=-1)), ("5%", np.percentile(samples, 5, axis=-1)),
                            ("10%", np.percentile(samples, 10, axis=-1)), ("p25", np.percentile(samples, 25, axis=-1)),
                            ("p50", np.percentile(samples, 50, axis=-1)), ("p75", np.percentile(samples, 75, axis=-1)),
                            ("p90", np.percentile(samples, 90, axis=-1)), ("p95", np.percentile(samples, 95, axis=-1)),
                            ("p99", np.percentile(samples, 99, axis=-1))])

    def generate_samples(self, stats=False, parameter=(), **kwargs):
        samples = self.sample(parameter, **kwargs)
        self.stats = self.compute_stats(samples)
        if stats:
            return samples, self.stats
        else:
            return samples
