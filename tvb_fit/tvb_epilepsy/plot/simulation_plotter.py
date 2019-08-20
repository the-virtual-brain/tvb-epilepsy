# coding=utf-8

from tvb_fit.base.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)

import numpy

from tvb_fit.tvb_epilepsy.base.model.epileptor_models import EpileptorDP2D, EpileptorDPrealistic
from tvb_fit.tvb_epilepsy.base.model.timeseries import TimeseriesDimensions, PossibleVariables

from tvb_timeseries.service.timeseries_service import TimeseriesService
from tvb_plot.timeseries_plotter import TimeseriesPlotter


class SimulationPlotter(TimeseriesPlotter):

    def __init__(self, config=None):
        super(SimulationPlotter, self).__init__(config)

    def plot_simulated_seeg_timeseries(self, seeg_dict, title_prefix="Ep"):
        figs = []
        for sensors_name, seeg in seeg_dict.items():
            title = title_prefix + "Simulated " + sensors_name + " raster plot"
            figs.append(self.plot_raster(seeg, title=title, offset=0.1, figsize=FiguresConfig.VERY_LARGE_SIZE))
        return tuple(figs)

    def plot_simulated_timeseries(self, timeseries, model, seizure_indices, seeg_dict={},
                                  spectral_raster_plot=False, title_prefix="", spectral_options={}):
        figs = []
        if len(title_prefix) > 0:
            title_prefix = title_prefix + ", " + model._ui_name + ": "
        source_ts = timeseries.get_source()
        start_plot = int(numpy.round(0.01 * source_ts.data.shape[0]))
        figs.append(self.plot_raster(source_ts.get_time_window(start_plot, -1), special_idx=seizure_indices,
                                     title=title_prefix + "Simulated source rasterplot", offset=0.1,
                                     figsize=FiguresConfig.VERY_LARGE_SIZE))

        if isinstance(model, EpileptorDP2D):
            # We assume that at least x1 and z are available in res
            this_ts = timeseries.get_variables(['x1', 'z'])
            figs.append(self.plot_timeseries(this_ts, special_idx=seizure_indices,
                                             title=title_prefix + "Simulated TAVG",
                                             figsize=FiguresConfig.VERY_LARGE_SIZE))

            figs.append(self.plot_trajectories(this_ts, special_idx=seizure_indices,
                                               title=title_prefix + 'Simulated state space trajectories',
                                               figsize=FiguresConfig.LARGE_SIZE))
        else:
            state_variables = timeseries.labels_dimensions[TimeseriesDimensions.VARIABLES.value]
            # We assume that at least source and z are available in res
            source_ts = timeseries.get_source()
            figs.append(self.plot_timeseries(TimeseriesService().concatenate_variables([source_ts, timeseries.z]),
                                             special_idx=seizure_indices,
                                             title=title_prefix + "Simulated source-z",
                                             figsize=FiguresConfig.VERY_LARGE_SIZE))

            if PossibleVariables.X1.value in state_variables and PossibleVariables.Y1.value in state_variables:
                figs.append(self.plot_timeseries(timeseries.get_variables(['x1', 'y1']),
                                                 special_idx=seizure_indices, title=title_prefix + "Simulated pop1",
                                                 figsize=FiguresConfig.VERY_LARGE_SIZE))
            if PossibleVariables.X2.value in state_variables and PossibleVariables.Y2.value in state_variables and \
                    PossibleVariables.G.value in state_variables:
                figs.append(self.plot_timeseries(timeseries.get_variables(['x2', 'y2', "g"]),
                                                 special_idx=seizure_indices, title=title_prefix + "Simulated pop2-g",
                                                 figsize=FiguresConfig.VERY_LARGE_SIZE))

            if spectral_raster_plot:
                figs.append(self.plot_spectral_analysis_raster(source_ts, freq=None,
                                                               spectral_options=spectral_options,
                                                               special_idx=seizure_indices,
                                                               title=title_prefix + "Simulated Spectral Analysis",
                                                               figsize=FiguresConfig.LARGE_SIZE))

            if isinstance(model, EpileptorDPrealistic):
                if PossibleVariables.X0_T.value in state_variables \
                        and PossibleVariables.IEXT1_T.value in state_variables \
                        and PossibleVariables.K_T.value:
                    figs.append(self.plot_timeseries(timeseries.get_variables(['x0_values', 'Iext1', "K"]),
                                                     special_idx=seizure_indices,
                                                     title=title_prefix + "Simulated parameters",
                                                     figsize=FiguresConfig.VERY_LARGE_SIZE))
                if PossibleVariables.SLOPE_T.value in state_variables and \
                        PossibleVariables.IEXT2_T.value in state_variables:
                    title = model._ui_name + ": Simulated controlled parameters"
                    figs.append(self.plot_timeseries(timeseries.get_variables(['z', 'slope', "Iext2"]),
                                                     special_idx=seizure_indices, title=title_prefix + title,
                                                     figsize=FiguresConfig.VERY_LARGE_SIZE))

        figs.append(self.plot_simulated_seeg_timeseries(seeg_dict, title_prefix=title_prefix))

        return tuple(figs)
