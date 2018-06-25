import importlib
import numpy as np
from SALib.sample import saltelli, fast_sampler, morris, ff
from tvb_infer.base.config import CalculusConfig
from tvb_infer.base.utils.log_error_utils import raise_not_implemented_error
from tvb_infer.base.model.parameter import Parameter
from tvb_infer.samplers.probabilistic_sampler import ProbabilisticSampler


class SalibSamplerInterface(ProbabilisticSampler):

    def __init__(self, n_samples=10, sampler="saltelli", random_seed=None):
        super(SalibSamplerInterface, self).__init__(n_samples, "salib", random_seed)
        self.sampling_module = "SALib"
        self.sampler = sampler.lower()

    def sample(self, parameter=(), loc=0.0, scale=1.0, **kwargs):
        if isinstance(parameter, Parameter):
            parameter_shape = parameter.p_shape
            low = parameter.low
            high = parameter.high
            loc = getattr(parameter, "loc", loc)
            scale = getattr(parameter, "scale", scale)
        else:
            low = np.array(kwargs.pop("low", -CalculusConfig.MAX_SINGLE_VALUE))
            high = np.array(kwargs.pop("high", CalculusConfig.MAX_SINGLE_VALUE))
            parameter_shape = kwargs.pop("shape", (1,))
        scale = (high - low) * scale
        low = low + loc
        high = low + scale
        low, high = self.check_for_infinite_bounds(low.tolist(), high.tolist())
        low, high, n_outputs, parameter_shape = self.check_size(low, high, parameter_shape)
        bounds = [list(b) for b in zip(low.tolist(), high.tolist())]
        self.adjust_shape(parameter_shape)
        self.sampler = importlib.import_module("SALib.sample." + self.sampler).sample
        size = self.n_samples
        problem = {'num_vars': n_outputs, 'bounds': bounds}
        if self.sampler is ff.sample:
            samples = (self.sampler(problem)).T
        else:
            other_params = {}
            if self.sampler is saltelli.sample:
                size = int(np.round(1.0 * size / (2 * n_outputs + 2)))
            elif self.sampler is fast_sampler.sample:
                other_params = {"M": kwargs.get("M", 4)}
            elif self.sampler is morris.sample:
                # I don't understand this method and its inputs. I don't think we will ever use it.
                raise_not_implemented_error()
            samples = self.sampler(problem, size, **other_params)
        # Adjust samples number:
        self.n_samples = samples.shape[0]
        self.shape = list(self.shape)
        self.shape[-1] = self.n_samples
        self.shape = tuple(self.shape)
        transpose_shape = tuple([self.n_samples] + list(self.shape)[0:-1])
        return np.reshape(samples.T, transpose_shape).T
