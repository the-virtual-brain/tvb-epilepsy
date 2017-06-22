import numpy as np
import numpy.random as nr
import scipy.stats as ss
import pystan as ps


def mean_std_to_low_high(mu=0.0, std=1.0):

    std = std * np.sqrt(3.0)

    low = mu - std
    high = mu + std

    return low, high


def low_high_to_mean_std(low=0.0, high=1.0):

    mu = (low + high) / 2.0

    std = (high - low) / 2.0 / np.sqrt(3)

    return mu, std

# def loc_scale_to_mean_std(loc=0.0, scale=1.0, distribution=):


class Sample_Service(object):

    def __init__(self, sampling_type= "deterministic"):

        if sampling_type == "deterministic":

            self.sample_engine = np

        elif sampling_type == "numpy":

            self.sample_engine = nr

        elif sampling_type == "scipy":

            self.sample_engine = ss

        elif sampling_type == "pystan":

            self.sample_engine = ps

        else:

            raise ValueError("Sampler engine type "+ str(type) + " is not recognized!")



    def gen_samples(self, distribution):
        if self.stoch_sample_engine == nr or self.stoch_sample_engine == ss:
            return getattr(self.stoch_sample_engine, distribution)
        # elif self.sample_engine == ps:
        #     return pystan_sample()


    def sample(self, shape=(1,), distribution="linspace", **kwargs):

        size = np.array(shape).size

        return np.reshape(self.gen_samples(distribution, kwargs), shape)


