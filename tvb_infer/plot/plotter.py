# coding=utf-8

from tvb_infer.plot.head_plotter import HeadPlotter
from tvb_infer.plot.timeseries_plotter import TimeseriesPlotter
from tvb_infer.plot.probabilistic_plotter import ProbabilisticPlotter
from tvb_infer.plot.stan_plotter import STANplotter
from tvb_infer.tvb_lsa.lsa_plotter import LSAPlotter


class Plotter(object):

    def __init__(self, config=None):
        self.config = config

    def plot_head(self, head):
        return HeadPlotter(self.config).plot_head(head)

    def plot_timeseries(self, *args, **kwargs):
        return TimeseriesPlotter(self.config).plot_timeseries(*args, **kwargs)

    def plot_raster(self,  *args, **kwargs):
        return TimeseriesPlotter(self.config).plot_raster( *args, **kwargs)

    def plot_trajectories(self, *args, **kwargs):
        return TimeseriesPlotter(self.config).plot_timeseries( *args, **kwargs)

    def plot_spectral_analysis_raster(self, *args, **kwargs):
        return TimeseriesPlotter(self.config).plot_spectral_analysis_raster(self, *args, **kwargs)

    def plot_distribution(self, *args, ** kwargs):
        return ProbabilisticPlotter(self.config).plot_distribution(*args, **kwargs)

    def plot_probabilistic_parameter(self, *args, **kwargs):
        return ProbabilisticPlotter(self.config).plot_probabilistic_parameter(*args, **kwargs)

    def plot_probabilistic_model(self, *args, **kwargs):
        return ProbabilisticPlotter(self.config).plot_probabilistic_model(*args, **kwargs)

    def plot_HMC(self, *args, **kwargs):
        STANplotter(self.config).plot_HMC(*args, **kwargs)

    def plot_lsa_eigen_vals_vectors(self, *args, **kwargs):
        LSAPlotter(self.config).plot_lsa(*args, **kwargs)

    def plot_lsa(self, *args, **kwargs):
        LSAPlotter(self.config).plot_lsa(*args, **kwargs)