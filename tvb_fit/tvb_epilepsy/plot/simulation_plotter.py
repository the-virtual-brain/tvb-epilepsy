# coding=utf-8

from tvb_fit.base.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)

import numpy

from tvb_fit.base.utils.data_structures_utils import ensure_list
from tvb_fit.tvb_epilepsy.base.model.epileptor_models import EpileptorDP2D, EpileptorDPrealistic
from tvb_fit.tvb_epilepsy.base.model.timeseries import TimeseriesDimensions, PossibleVariables
from tvb_fit.plot.timeseries_plotter import TimeseriesPlotter


class SimulationPlotter(TimeseriesPlotter):

    def __init__(self, config=None):
        super(SimulationPlotter, self).__init__(config)

    def plot_simulated_seeg_timeseries(self, seeg_dict, title_prefix="Ep"):
        figs = []
        for sensors_name, seeg in seeg_dict.items():
            title = title_prefix + "Simulated " +  sensors_name + " raster plot"
            figs.append(self.plot_raster({'SEEG': seeg.squeezed}, seeg.time_line,
                                         time_units=seeg.time_unit, title=title, offset=0.1, labels=seeg.space_labels,
                                         figsize=FiguresConfig.VERY_LARGE_SIZE))
        return tuple(figs)

    def plot_simulated_timeseries(self, timeseries, model, seizure_indices, seeg_dict={},
                                  spectral_raster_plot=False, title_prefix="", spectral_options={}):
        figs = []
        if len(title_prefix) > 0:
            title_prefix = title_prefix + ", " + model._ui_name + ": "
        region_labels = timeseries.space_labels
        state_variables = timeseries.dimension_labels[TimeseriesDimensions.VARIABLES.value]
        source_ts = timeseries.get_source()
        start_plot = int(numpy.round(0.01 * source_ts.data.shape[0]))
        figs.append(self.plot_raster({'source(t)': source_ts.squeezed[start_plot:, :]},
                                     timeseries.time_line.flatten()[start_plot:],
                                     time_units=timeseries.time_unit, special_idx=seizure_indices,
                                     title=title_prefix + "Simulated source rasterplot", offset=0.1,
                                     labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE))

        if isinstance(model, EpileptorDP2D):
            # We assume that at least x1 and z are available in res
            sv_dict = {'x1(t)': timeseries.x1.squeezed, 'z(t)': timeseries.z.squeezed}

            figs.append(self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                 special_idx=seizure_indices, title=title_prefix + "Simulated TAVG",
                                 labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE))

            figs.append(self.plot_trajectories(sv_dict, special_idx=seizure_indices,
                                               title=title_prefix + 'Simulated state space trajectories',
                                               labels=region_labels,
                                               figsize=FiguresConfig.LARGE_SIZE))
        else:
            # We assume that at least source and z are available in res
            sv_dict = {'source(t)': source_ts.squeezed, 'z(t)': timeseries.z.squeezed}

            figs.append(self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                             special_idx=seizure_indices, title=title_prefix + "Simulated source-z",
                                             labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE))

            if PossibleVariables.X1.value in state_variables and PossibleVariables.Y1.value in state_variables:
                sv_dict = {'x1(t)': timeseries.x1.squeezed, 'y1(t)': timeseries.y1.squeezed}

                figs.append(self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                                 special_idx=seizure_indices, title=title_prefix + "Simulated pop1",
                                                 labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE))
            if PossibleVariables.X2.value in state_variables and PossibleVariables.Y2.value in state_variables and \
                    PossibleVariables.G.value in state_variables:
                sv_dict = {'x2(t)': timeseries.x2.squeezed, 'y2(t)': timeseries.y2.squeezed,
                           'g(t)': timeseries.g.squeezed}

                figs.append(self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                                 special_idx=seizure_indices, title=title_prefix + "Simulated pop2-g",
                                                 labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE))

            if spectral_raster_plot:
                figs.append(self.plot_spectral_analysis_raster(timeseries.time_line, source_ts.squeezed,
                                                               time_units=timeseries.time_unit, freq=None,
                                                               spectral_options=spectral_options,
                                                               special_idx=seizure_indices,
                                                               title=title_prefix + "Simulated Spectral Analysis",
                                                               labels=region_labels, figsize=FiguresConfig.LARGE_SIZE))

            if isinstance(model, EpileptorDPrealistic):
                if PossibleVariables.SLOPE_T.value in state_variables and \
                        PossibleVariables.IEXT2_T.value in state_variables:
                    sv_dict = {'1/(1+exp(-10(z-3.03))': 1 / (1 + numpy.exp(-10 * (timeseries.z.squeezed - 3.03))),
                               'slope': timeseries.slope_t.squeezed, 'Iext2': timeseries.Iext2_t.squeezed}
                    title = model._ui_name + ": Simulated controlled parameters"

                    figs.append(self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                                     special_idx=seizure_indices, title=title_prefix + title,
                                                     labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE))
                if PossibleVariables.X0_T.value in state_variables and PossibleVariables.IEXT1_T.value in state_variables \
                        and PossibleVariables.K_T.value:
                    sv_dict = {'x0_values': timeseries.x0_t.squeezed, 'Iext1': timeseries.Iext1_t.squeezed,
                               'K': timeseries.K_t.squeezed}

                    figs.append(self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                                     special_idx=seizure_indices,
                                                     title=title_prefix + "Simulated parameters",
                                                     labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE))

        figs.append(self.plot_simulated_seeg_timeseries(seeg_dict, title_prefix=title_prefix))

        return tuple(figs)
