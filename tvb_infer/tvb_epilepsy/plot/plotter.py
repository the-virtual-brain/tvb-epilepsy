from tvb_infer.plot.plotter import Plotter as PlotterBase
from tvb_infer.tvb_epilepsy.plot.model_config_plotter import ModelConfigPlotter
from tvb_infer.tvb_epilepsy.plot.lsa_plotter import LSAPlotter
from tvb_infer.tvb_epilepsy.plot.model_inversion_plotter import ModelInversionPlotter
from tvb_infer.tvb_epilepsy.plot.simulation_plotter import SimulationPlotter


class Plotter(PlotterBase):

    def __init__(self, config=None):
        super(Plotter, self).__init__(config)

    def plot_state_space(self, *args, **kwargs):
        return ModelConfigPlotter(self.config).plot_state_space(*args, **kwargs)

    def plot_model_configuration(self, *args, **kwargs):
        return ModelConfigPlotter(self.config).plot_model_configuration(*args, **kwargs)

    def plot_lsa_eigen_vals_vectors(self, *args, **kwargs):
        return LSAPlotter(self.config).plot_lsa(*args, **kwargs)

    def plot_lsa(self, *args, **kwargs):
        return LSAPlotter(self.config).plot_lsa(*args, **kwargs)

    def plot_simulated_seeg_timeseries(self, *args, **kwargs):
        return SimulationPlotter(self.config).plot_simulated_seeg_timeseries(self, *args, **kwargs)

    def plot_simulated_timeseries(self, *args, **kwargs):
        return SimulationPlotter(self.config).plot_simulated_timeseries(*args, **kwargs)

    def plot_fit_scalar_params_iters(self, *args, **kwargs):
        return ModelInversionPlotter(self.config).plot_fit_scalar_params_iters(*args, **kwargs)

    def plot_fit_scalar_params(self, *args, **kwargs):
        return ModelInversionPlotter(self.config).plot_fit_scalar_params(*args, **kwargs)

    def plot_fit_region_params(self, *args, **kwargs):
        return ModelInversionPlotter(self.config).plot_fit_region_params(*args, **kwargs)

    def plot_fit_timeseries(self, *args, **kwargs):
        return ModelInversionPlotter(self.config).plot_fit_timeseries(*args, **kwargs)

    def plot_fit_connectivity(self, *args, **kwargs):
        return ModelInversionPlotter(self.config).plot_fit_connectivity(*args, **kwargs)

    def plot_scalar_model_comparison(self, *args, **kwargs):
        return ModelInversionPlotter(self.config).plot_scalar_model_comparison(*args, **kwargs)

    def plot_array_model_comparison(self, *args, **kwargs):
        return ModelInversionPlotter(self.config).plot_array_model_comparison(*args, **kwargs)

    def plot_fit_results(self, *args, **kwargs):
        return ModelInversionPlotter(self.config).plot_fit_results(*args, **kwargs)