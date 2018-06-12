# coding=utf-8

from tvb_epilepsy.plot.head_plotter import HeadPlotter
from tvb_epilepsy.plot.model_config_plotter import ModelConfigPlotter
from tvb_epilepsy.plot.lsa_plotter import LSAPlotter
from tvb_epilepsy.plot.timeseries_plotter import TimeseriesPlotter
from tvb_epilepsy.plot.simulation_plotter import SimulationPlotter
from tvb_epilepsy.plot.probabilistic_plotter import ProbabilisticPlotter
from tvb_epilepsy.plot.model_inversion_plotter import ModelInversionPlotter


class Plotter(object):

    def __init__(self, config=None):
        self.config = config

    def plot_head(self, head):
        return HeadPlotter(self.config).plot_head(head)

    def plot_state_space(self, *args, **kwargs):
        return ModelConfigPlotter(self.config).plot_state_space(*args, **kwargs)

    def plot_model_configuration(self, *args, **kwargs):
        return ModelConfigPlotter(self.config).plot_model_configuration(*args, **kwargs)

    def plot_lsa_eigen_vals_vectors(self, *args, **kwargs):
        return LSAPlotter(self.config).plot_lsa_eigen_vals_vectors(*args, **kwargs)

    def plot_lsa(self, *args, **kwargs):
        return LSAPlotter(self.config).plot_lsa(*args, **kwargs)

    def plot_timeseries(self, *args, **kwargs):
        return TimeseriesPlotter(self.config).plot_timeseries(*args, **kwargs)

    def plot_raster(self,  *args, **kwargs):
        return TimeseriesPlotter(self.config).plot_raster( *args, **kwargs)

    def plot_trajectories(self, *args, **kwargs):
        return TimeseriesPlotter(self.config).plot_timeseries( *args, **kwargs)

    def plot_spectral_analysis_raster(self, *args, **kwargs):
        return TimeseriesPlotter(self.config).plot_spectral_analysis_raster(self, *args, **kwargs)

    def plot_simulated_seeg_timeseries(self, *args, **kwargs):
        return SimulationPlotter(self.config).plot_simulated_seeg_timeseries(self, *args, **kwargs)

    def plot_simulated_timeseries(self, *args, **kwargs):
        return SimulationPlotter(self.config).plot_simulated_timeseries(*args, **kwargs)

    def plot_distribution(self, *args, ** kwargs):
        return ProbabilisticPlotter(self.config).plot_distribution(*args, **kwargs)

    def plot_probabilistic_parameter(self, *args, **kwargs):
        return ProbabilisticPlotter(self.config).plot_probabilistic_parameter(*args, **kwargs)

    def plot_probabilistic_model(self, *args, **kwargs):
        return ProbabilisticPlotter(self.config).plot_probabilistic_model(*args, **kwargs)

    def plot_HMC(self, *args, **kwargs):
        ModelInversionPlotter(self.config).plot_HMC(*args, **kwargs)

    def plot_fit_scalar_params_iters(self, *args, **kwargs):
        ModelInversionPlotter(self.config).plot_fit_scalar_params_iters(*args, **kwargs)

    def plot_fit_scalar_params(self, *args, **kwargs):
        ModelInversionPlotter(self.config).plot_fit_scalar_params(*args, **kwargs)

    def plot_fit_region_params(self, *args, **kwargs):
        ModelInversionPlotter(self.config).plot_fit_region_params(*args, **kwargs)

    def plot_fit_timeseries(self, *args, **kwargs):
        ModelInversionPlotter(self.config).plot_fit_timeseries(*args, **kwargs)

    def plot_fit_connectivity(self, *args, **kwargs):
        ModelInversionPlotter(self.config).plot_fit_connectivity(*args, **kwargs)

    def plot_scalar_model_comparison(self, *args, **kwargs):
        ModelInversionPlotter(self.config).plot_scalar_model_comparison(*args, **kwargs)

    def plot_array_model_comparison(self, *args, **kwargs):
        ModelInversionPlotter(self.config).plot_array_model_comparison(*args, **kwargs)

    def plot_fit_results(self, *args, **kwargs):
        ModelInversionPlotter(self.config).plot_fit_results(*args, **kwargs)

