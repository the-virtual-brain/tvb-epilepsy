import numpy as np
from tvb_fit.base.config import CalculusConfig
from tvb_fit.base.model.parameter import Parameter
from tvb_fit.samplers.sampler_base import SamplerBase


class DeterministicSampler(SamplerBase):

    def __init__(self, n_samples=10, grid_mode=True):
        super(DeterministicSampler, self).__init__(n_samples)
        self.sampling_module = "numpy.linspace"
        self.sampler = np.linspace
        self.grid_mode = grid_mode
        self.shape = (1, self.n_samples)

    def sample(self, parameter=(), **kwargs):
        if isinstance(parameter, Parameter):
            parameter_shape = parameter.shape
            low = parameter.low
            high = parameter.high
        else:
            parameter_shape = kwargs.pop("shape", (1,))
            low = kwargs.pop("low", -CalculusConfig.MAX_SINGLE_VALUE)
            high = kwargs.pop("high", CalculusConfig.MAX_SINGLE_VALUE)
        low, high = self.check_for_infinite_bounds(low, high)
        low, high, n_outputs, parameter_shape = self.check_size(low, high, parameter_shape)
        self.adjust_shape(parameter_shape)
        samples = []
        for (lo, hi) in zip(low.flatten(), high.flatten()):
            samples.append(self.sampler(lo, hi, self.n_samples))
        if self.grid_mode:
            samples_grids = np.meshgrid(*(samples), sparse=False, indexing="ij")
            samples = []
            for sb in samples_grids:
                samples.append(sb.flatten())
            samples = np.array(samples)
            self.shape = samples.shape
            self.n_samples = self.shape[1]
        else:
            samples = np.array(samples)
        transpose_shape = tuple([self.n_samples] + list(self.shape)[0:-1])
        samples = np.reshape(samples, transpose_shape).T
        self.shape = samples.shape
        return samples
