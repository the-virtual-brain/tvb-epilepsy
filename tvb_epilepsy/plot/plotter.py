# coding=utf-8

from tvb_epilepsy.base.constants.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig.MATPLOTLIB_BACKEND)
from matplotlib import pyplot, gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy
from collections import OrderedDict

from tvb_epilepsy.base.constants.model_constants import TAU0_DEF, TAU1_DEF, X1EQ_CR_DEF, X1_DEF, X0_CR_DEF, X0_DEF
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list, isequal_string, sort_dict, linspace_broadcast, \
    generate_region_labels, ensure_string, list_of_dicts_to_dicts_of_ndarrays, extract_dict_stringkeys
from tvb_epilepsy.base.computations.math_utils import compute_in_degree
from tvb_epilepsy.base.computations.calculations_utils import calc_fz, calc_fx1, calc_fx1_2d_taylor, \
                                                                            calc_x0_val_to_model_x0, raise_value_error
from tvb_epilepsy.base.computations.equilibrium_computation import calc_eq_y1, def_x1lin
from tvb_epilepsy.base.computations.analyzers_utils import time_spectral_analysis
from tvb_epilepsy.base.epileptor_models import EpileptorDP2D, EpileptorDPrealistic
from tvb_epilepsy.base.model.vep.sensors import Sensors, SensorTypes
from tvb_epilepsy.base.model.timeseries import TimeseriesDimensions, PossibleVariables
from tvb_epilepsy.plot.base_plotter import BasePlotter


class Plotter(BasePlotter):

    def __init__(self, config=None):
        super(Plotter, self).__init__(config)
        self.HighlightingDataCursor = lambda *args, **kwargs: None
        if matplotlib.get_backend() in matplotlib.rcsetup.interactive_bk and self.config.figures.MOUSE_HOOVER:
            try:
                from mpldatacursor import HighlightingDataCursor
                self.HighlightingDataCursor = HighlightingDataCursor
            except ImportError:
                self.config.figures.MOUSE_HOOVER = False
                self.logger.warning("Importing mpldatacursor failed! No highlighting functionality in plots!")
        else:
            self.logger.warning("Noninteractive matplotlib backend! No highlighting functionality in plots!")
            self.config.figures.MOUSE_HOOVER = False

    def _plot_connectivity(self, connectivity, figure_name='Connectivity '):
        pyplot.figure(figure_name + str(connectivity.number_of_regions), self.config.figures.VERY_LARGE_SIZE)
        self.plot_regions2regions(connectivity.normalized_weights, connectivity.region_labels, 121,
                                  "normalised weights")
        self.plot_regions2regions(connectivity.tract_lengths, connectivity.region_labels, 122, "tract lengths")
        self._save_figure(None, figure_name.replace(" ", "_").replace("\t", "_"))
        self._check_show()

    def _plot_connectivity_stats(self, connectivity, figsize=FiguresConfig.VERY_LARGE_SIZE, figure_name='HeadStats '):
        pyplot.figure("Head stats " + str(connectivity.number_of_regions), figsize=figsize)
        areas_flag = len(connectivity.areas) == len(connectivity.region_labels)
        ax = self.plot_vector(compute_in_degree(connectivity.normalized_weights), connectivity.region_labels,
                              111 + 10 * areas_flag,
                              "w in-degree")
        ax.invert_yaxis()
        if len(connectivity.areas) == len(connectivity.region_labels):
            ax = self.plot_vector(connectivity.areas, connectivity.region_labels, 122, "region areas")
            ax.invert_yaxis()
        self._save_figure(None, figure_name.replace(" ", "").replace("\t", ""))
        self._check_show()

    def _plot_sensors(self, sensors, region_labels, count=1):
        if sensors.gain_matrix is None:
            return count
        self._plot_gain_matrix(sensors, region_labels, title=str(count) + " - " + sensors.s_type + " - Projection")
        count += 1
        return count

    def _plot_gain_matrix(self, sensors, region_labels, figure=None, title="Projection", y_labels=1, x_labels=1,
                          x_ticks=numpy.array([]), y_ticks=numpy.array([]), figsize=FiguresConfig.VERY_LARGE_SIZE):
        if not (isinstance(figure, pyplot.Figure)):
            figure = pyplot.figure(title, figsize=figsize)
        n_sensors = sensors.number_of_sensors
        number_of_regions = len(region_labels)
        if len(x_ticks) == 0:
            x_ticks = numpy.array(range(n_sensors), dtype=numpy.int32)
        if len(y_ticks) == 0:
            y_ticks = numpy.array(range(number_of_regions), dtype=numpy.int32)
        cmap = pyplot.set_cmap('autumn_r')
        img = pyplot.imshow(sensors.gain_matrix[x_ticks][:, y_ticks].T, cmap=cmap, interpolation='none')
        pyplot.grid(True, color='black')
        if y_labels > 0:
            # region_labels = numpy.array(["%d. %s" % l for l in zip(range(number_of_regions), region_labels)])
            region_labels = generate_region_labels(number_of_regions, region_labels, ". ")
            pyplot.yticks(y_ticks, region_labels[y_ticks])
        else:
            pyplot.yticks(y_ticks)
        if x_labels > 0:
            sensor_labels = numpy.array(["%d. %s" % l for l in zip(range(n_sensors), sensors.labels)])
            pyplot.xticks(x_ticks, sensor_labels[x_ticks], rotation=90)
        else:
            pyplot.xticks(x_ticks)
        ax = figure.get_axes()[0]
        ax.autoscale(tight=True)
        pyplot.title(title)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax1)  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0
        self._save_figure(None, title)
        self._check_show()
        return figure

    def plot_head(self, head):
        self._plot_connectivity(head.connectivity)
        self._plot_connectivity_stats(head.connectivity)
        count = 1
        for s_type in SensorTypes:
            sensors = getattr(head, "sensors" + s_type.value)
            if isinstance(sensors, (list, Sensors)):
                sensors_list = ensure_list(sensors)
                if len(sensors_list) > 0:
                    for s in sensors_list:
                        count = self._plot_sensors(s, head.connectivity.region_labels, count)

    def plot_model_configuration(self, model_configuration, number_of_regions=None, regions_labels=[], x0_indices=[],
                                 e_indices=[], disease_indices=[], title="Model Configuration Overview", figure_name='',
                                 figsize=FiguresConfig.VERY_LARGE_SIZE):
        if number_of_regions is None:
            number_of_regions = len(model_configuration.x0_values)
        if not regions_labels:
            regions_labels = numpy.array([str(ii) for ii in range(number_of_regions)])
        disease_indices = numpy.unique(numpy.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()
        plot_dict_list = model_configuration.prepare_for_plot(x0_indices, e_indices, disease_indices)
        return self.plot_in_columns(plot_dict_list, regions_labels, width_ratios=[],
                                    left_ax_focus_indices=disease_indices, right_ax_focus_indices=disease_indices,
                                    title=title, figure_name=figure_name, figsize=figsize)

    def plot_probabilistic_model(self, probabilistic_model, figure_name=""):
        _, ax = pyplot.subplots(len(probabilistic_model.parameters), 1, figsize=FiguresConfig.VERY_LARGE_PORTRAIT)
        for ip, p in enumerate(probabilistic_model.parameters.values()):
            self._prepare_parameter_axes(p, x=numpy.array([]), ax=ax[ip], lgnd=False)
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return ax, pyplot.gcf()

    def _timeseries_plot(self, time, n_vars, nTS, n_times, time_units, subplots, offset=0.0, data_lims=[]):
        def_time = range(n_times)
        try:
            time = numpy.array(time).flatten()
            if len(time) != n_times:
                self.logger.warning("Input time doesn't match data! Setting a default time step vector!")
                time = def_time
        except:
            self.logger.warning("Setting a default time step vector manually! Input time: " + str(time))
            time = def_time
        time_units = ensure_string(time_units)
        data_fun = lambda data, time, icol: (data[icol], time, icol)

        def plot_ts(x, iTS, colors, alphas, labels):
            x, time, ivar = x
            try:
                return pyplot.plot(time, x[:, iTS], colors[iTS], label=labels[iTS], alpha=alphas[iTS])
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(time, x[:, iTS], colors[iTS], label=str(iTS), alpha=alphas[iTS])

        def plot_ts_raster(x, iTS, colors, alphas, labels, offset):
            x, time, ivar = x
            try:
                return pyplot.plot(time, -x[:, iTS] + (offset * iTS + x[:, iTS].mean()), colors[iTS], label=labels[iTS],
                                   alpha=alphas[iTS])
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(time, -x[:, iTS] + offset * iTS, colors[iTS], label=str(iTS),
                                   alpha=alphas[iTS])

        def axlabels_ts(labels, n_rows, irow, iTS):
            if irow == n_rows:
                pyplot.gca().set_xlabel("Time (" + time_units+ ")")
            if n_rows > 1:
                try:
                    pyplot.gca().set_ylabel(str(iTS) + "." + labels[iTS])
                except:
                    self.logger.warning("Cannot convert labels' strings for y axis labels!")
                    pyplot.gca().set_ylabel(str(iTS))

        def axlimits_ts(data_lims, time, icol):
            pyplot.gca().set_xlim([time[0], time[-1]])
            if n_rows > 1:
                pyplot.gca().set_ylim([data_lims[icol][0], data_lims[icol][1]])
            else:
                pyplot.autoscale(enable=True, axis='y', tight=True)

        def axYticks(labels, nTS, ivar, offsets=offset):
            pyplot.gca().set_yticks((offset * numpy.array([range(nTS)]).flatten()).tolist())
            try:
                pyplot.gca().set_yticklabels(labels.flatten().tolist())
            except:
                labels = generate_region_labels(nTS, [], "")
                self.logger.warning("Cannot convert region labels' strings for y axis ticks!")

        if offset > 0.0:
            plot_lines = lambda x, iTS, colors, alphas, labels: plot_ts_raster(x, iTS, colors, alphas, labels, offset)
        else:
            plot_lines = lambda x, iTS, colors, alphas, labels: plot_ts(x, iTS, colors, alphas, labels)
        this_axYticks = lambda labels, nTS, ivar: axYticks(labels, nTS, offset)
        if subplots:
            n_rows = nTS
            def_alpha = 1.0
        else:
            n_rows = 1
            def_alpha = 0.5
        subtitle_col = lambda subtitle: pyplot.gca().set_title(subtitle)
        subtitle = lambda iTS, labels: None
        projection = None
        axlabels = lambda labels, vars, n_vars, n_rows, irow, iTS: axlabels_ts(labels, n_rows, irow, iTS)
        axlimits = lambda data_lims, time, n_vars, icol: axlimits_ts(data_lims, time, icol)
        loopfun = lambda nTS, n_rows, icol: range(nTS)
        return data_fun, time, plot_lines, projection, n_rows, n_vars, def_alpha, loopfun, \
               subtitle, subtitle_col, axlabels, axlimits, this_axYticks

    def _trajectories_plot(self, n_dims, nTS, nSamples, subplots):
        data_fun = lambda data, time, icol: data

        def plot_traj_2D(x, iTS, colors, alphas, labels):
            x, y = x
            try:
                return pyplot.plot(x[:, iTS], y[:, iTS], colors[iTS], label=labels[iTS], alpha=alphas[iTS])
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(x[:, iTS], y[:, iTS], colors[iTS], label=str(iTS), alpha=alphas[iTS])

        def plot_traj_3D(x, iTS, colors, alphas, labels):
            x, y, z = x
            try:
                return pyplot.plot(x[:, iTS], y[:, iTS], z[:, iTS], colors[iTS], label=labels[iTS], alpha=alphas[iTS])
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(x[:, iTS], y[:, iTS], z[:, iTS], colors[iTS], label=str(iTS), alpha=alphas[iTS])

        def subtitle_traj(labels, iTS):
            try:
                pyplot.gca().set_title(str(iTS) + "." + labels[iTS])
            except:
                self.logger.warning("Cannot convert labels' strings for subplot titles!")
                pyplot.gca().set_title(str(iTS))

        def axlabels_traj(vars, n_vars):
            pyplot.gca().set_xlabel(vars[0])
            pyplot.gca().set_ylabel(vars[1])
            if n_vars > 2:
                pyplot.gca().set_zlabel(vars[2])

        def axlimits_traj(data_lims, n_vars):
            pyplot.gca().set_xlim([data_lims[0][0], data_lims[0][1]])
            pyplot.gca().set_ylim([data_lims[1][0], data_lims[1][1]])
            if n_vars > 2:
                pyplot.gca().set_zlim([data_lims[2][0], data_lims[2][1]])

        if n_dims == 2:
            plot_lines = lambda x, iTS, colors, labels, alphas: plot_traj_2D(x, iTS, colors, labels, alphas)
            projection = None
        elif n_dims == 3:
            plot_lines = lambda x, iTS, colors, labels, alphas: plot_traj_3D(x, iTS, colors, labels, alphas)
            projection = '3d'
        else:
            raise_value_error("Data dimensions are neigher 2D nor 3D!, but " + str(n_dims) + "D!")
        n_rows = 1
        n_cols = 1
        if subplots is None:
            if nSamples > 1:
                n_rows = int(numpy.floor(numpy.sqrt(nTS)))
                n_cols = int(numpy.ceil((1.0 * nTS) / n_rows))
        elif isinstance(subplots, (list, tuple)):
            n_rows = subplots[0]
            n_cols = subplots[1]
            if n_rows * n_cols < nTS:
                raise_value_error("Not enough subplots for all time series:"
                                  "\nn_rows * n_cols = product(subplots) = product(" + str(subplots) + " = "
                                  + str(n_rows * n_cols) + "!")
        if n_rows * n_cols > 1:
            def_alpha = 0.5
            subtitle = lambda labels, iTS: subtitle_traj(labels, iTS)
            subtitle_col = lambda subtitles, icol: None
        else:
            def_alpha = 1.0
            subtitle = lambda: None
            subtitle_col = lambda subtitles, icol: pyplot.gca().set_title(pyplot.gcf().title)
        axlabels = lambda labels, vars, n_vars, n_rows, irow, iTS: axlabels_traj(vars, n_vars)
        axlimits = lambda data_lims, time, n_vars, icol: axlimits_traj(data_lims, n_vars)
        loopfun = lambda nTS, n_rows, icol: range(icol, nTS, n_rows)
        return data_fun, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
               subtitle, subtitle_col, axlabels, axlimits

    def plot_timeseries(self, data_dict, time=None, mode="ts", subplots=None, special_idx=[], subtitles=[],
                        offset=1.0, time_units="ms", title='Time series', figure_name=None, labels=[],
                        figsize=FiguresConfig.LARGE_SIZE):
        n_vars = len(data_dict)
        vars = data_dict.keys()
        data = data_dict.values()
        data_lims = []
        for id, d in enumerate(data):
            if isequal_string(mode, "raster"):
                drange = numpy.percentile(d.flatten(), 95) - numpy.percentile(d.flatten(), 5)
                data[id] = d / drange # zscore(d, axis=None)
            data_lims.append([d.min(), d.max()])
        data_shape = data[0].shape
        n_times, nTS = data_shape[:2]
        if len(data_shape) > 2:
            nSamples = data_shape[2]
        else:
            nSamples = 1
        if len(subtitles) == 0:
            subtitles = vars
        labels = generate_region_labels(nTS, labels)
        if isequal_string(mode, "traj"):
            data_fun, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
            subtitle, subtitle_col, axlabels, axlimits = self._trajectories_plot(n_vars, nTS, nSamples, subplots)
        else:
            if isequal_string(mode, "raster"):
                data_fun, time, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
                subtitle, subtitle_col, axlabels, axlimits, axYticks = \
                    self._timeseries_plot(time, n_vars, nTS, n_times, time_units, 0, offset, data_lims)

            else:
                data_fun, time, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
                subtitle, subtitle_col, axlabels, axlimits, axYticks = \
                    self._timeseries_plot(time, n_vars, nTS, n_times, time_units, ensure_list(subplots)[0])
        alpha_ratio = 1.0 / nSamples
        colors = numpy.array(['k'] * nTS)
        alphas = numpy.maximum(numpy.array([def_alpha] * nTS) * alpha_ratio, 0.1)
        colors[special_idx] = 'r'
        alphas[special_idx] = numpy.maximum(alpha_ratio, 0.1)
        lines = []
        pyplot.figure(title, figsize=figsize)
        pyplot.hold(True)
        axes = []
        for icol in range(n_cols):
            if n_rows == 1:
                # If there are no more rows, create axis, and set its limits, labels and possible subtitle
                axes += ensure_list(pyplot.subplot(n_rows, n_cols, icol + 1, projection=projection))
                axlimits(data_lims, time, n_vars, icol)
                axlabels(labels, vars, n_vars, n_rows, 1, 0)
                pyplot.gca().set_title(subtitles[icol])
            for iTS in loopfun(nTS, n_rows, icol):
                if n_rows > 1:
                    # If there are more rows, create axes, and set their limits, labels and possible subtitles
                    axes += ensure_list(pyplot.subplot(n_rows, n_cols, iTS + 1, projection=projection))
                    subtitle(labels, iTS)
                    axlimits(data_lims, time, n_vars, icol)
                    axlabels(labels, vars, n_vars, n_rows, (iTS % n_rows) + 1, iTS)
                lines += ensure_list(plot_lines(data_fun(data, time, icol), iTS, colors, alphas, labels))
            if isequal_string(mode, "raster"):  # set yticks as labels if this is a raster plot
                axYticks(labels, nTS, icol)
                pyplot.gca().invert_yaxis()

        if self.config.figures.MOUSE_HOOVER:
            for line in lines:
                self.HighlightingDataCursor(line, formatter='{label}'.format, bbox=dict(fc='white'),
                                            arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))

        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return pyplot.gcf(), axes, lines

    def plot_raster(self, data_dict, time, time_units="ms", special_idx=[], title='Raster plot', subtitles=[],
                    offset=1.0, figure_name=None, labels=[], figsize=FiguresConfig.VERY_LARGE_SIZE):
        return self.plot_timeseries(data_dict, time, "raster", None, special_idx, subtitles, offset, time_units, title,
                                    figure_name, labels, figsize)

    def plot_trajectories(self, data_dict, subtitles=None, special_idx=[], title='State space trajectories',
                          figure_name=None, labels=[], figsize=FiguresConfig.LARGE_SIZE):
        return self.plot_timeseries(data_dict, [], "traj", subtitles, special_idx, title=title, figure_name=figure_name,
                                    labels=labels, figsize=figsize)

    def plot_spectral_analysis_raster(self, time, data, time_units="ms", freq=None, special_idx=[],
                                      title='Spectral Analysis', figure_name=None, labels=[],
                                      figsize=FiguresConfig.VERY_LARGE_SIZE, **kwargs):
        nS = data.shape[1]
        n_special_idx = len(special_idx)
        if n_special_idx > 0:
            data = data[:, special_idx]
            nS = data.shape[1]
            if len(labels) > n_special_idx:
                labels = numpy.array([str(ilbl) + ". " + str(labels[ilbl]) for ilbl in special_idx])
            elif len(labels) == n_special_idx:
                labels = numpy.array([str(ilbl) + ". " + str(label) for ilbl, label in zip(special_idx, labels)])
            else:
                labels = numpy.array([str(ilbl) for ilbl in special_idx])
        else:
            if len(labels) != nS:
                labels = numpy.array([str(ilbl) for ilbl in range(nS)])
        if nS > 20:
            warning("It is not possible to plot spectral analysis plots for more than 20 signals!")
            return
        if not isinstance(time_units, basestring):
            time_units = list(time_units)[0]
        time_units = ensure_string(time_units)
        if time_units in ("ms", "msec"):
            fs = 1000.0
        else:
            fs = 1.0
        fs = fs / numpy.mean(numpy.diff(time))
        log_norm = kwargs.get("log_norm", False)
        mode = kwargs.get("mode", "psd")
        psd_label = mode
        if log_norm:
            psd_label = "log" + psd_label
        stf, time, freq, psd = time_spectral_analysis(data, fs,
                                                      freq=freq,
                                                      mode=mode,
                                                      nfft=kwargs.get("nfft"),
                                                      window=kwargs.get("window", 'hanning'),
                                                      nperseg=kwargs.get("nperseg", int(numpy.round(fs / 4))),
                                                      detrend=kwargs.get("detrend", 'constant'),
                                                      noverlap=kwargs.get("noverlap"),
                                                      f_low=kwargs.get("f_low", 10.0),
                                                      log_scale=kwargs.get("log_scale", False))
        min_val = numpy.min(stf.flatten())
        max_val = numpy.max(stf.flatten())
        if nS > 2:
            figsize = FiguresConfig.VERY_LARGE_SIZE
        fig = pyplot.figure(title, figsize=figsize)
        fig.suptitle(title)
        gs = gridspec.GridSpec(nS, 23)
        ax = numpy.empty((nS, 2), dtype="O")
        img = numpy.empty((nS,), dtype="O")
        line = numpy.empty((nS,), dtype="O")
        for iS in range(nS, -1, -1):
            if iS < nS - 1:
                ax[iS, 0] = pyplot.subplot(gs[iS, :20], sharex=ax[iS, 0])
                ax[iS, 1] = pyplot.subplot(gs[iS, 20:22], sharex=ax[iS, 1], sharey=ax[iS, 0])
            else:
                ax[iS, 0] = pyplot.subplot(gs[iS, :20])
                ax[iS, 1] = pyplot.subplot(gs[iS, 20:22], sharey=ax[iS, 0])
            img[iS] = ax[iS, 0].imshow(numpy.squeeze(stf[:, :, iS]).T, cmap=pyplot.set_cmap('jet'),
                                       interpolation='none',
                                       norm=Normalize(vmin=min_val, vmax=max_val), aspect='auto', origin='lower',
                                       extent=(time.min(), time.max(), freq.min(), freq.max()))
            # img[iS].clim(min_val, max_val)
            ax[iS, 0].set_title(labels[iS])
            ax[iS, 0].set_ylabel("Frequency (Hz)")
            line[iS] = ax[iS, 1].plot(psd[:, iS], freq, 'k', label=labels[iS])
            pyplot.setp(ax[iS, 1].get_yticklabels(), visible=False)
            # ax[iS, 1].yaxis.tick_right()
            # ax[iS, 1].yaxis.set_ticks_position('both')
            if iS == (nS - 1):
                ax[iS, 0].set_xlabel("Time (" + time_units + ")")

                ax[iS, 1].set_xlabel(psd_label)
            else:
                pyplot.setp(ax[iS, 0].get_xticklabels(), visible=False)
            pyplot.setp(ax[iS, 1].get_xticklabels(), visible=False)
            ax[iS, 0].autoscale(tight=True)
            ax[iS, 1].autoscale(tight=True)
        # make a color bar
        cax = pyplot.subplot(gs[:, 22])
        pyplot.colorbar(img[0], cax=pyplot.subplot(gs[:, 22]))  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0
        cax.set_title(psd_label)
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return fig, ax, img, line, time, freq, stf, psd

    def plot_simulated_seeg_timeseries(self, seeg_list, title_prefix="Ep"):
        for seeg in ensure_list(seeg_list):
            title = title_prefix + "Simulated SEEG" + str(len(seeg.space_labels)) + " raster plot"
            self.plot_raster({'SEEG': seeg.squeezed}, seeg.time_line,
                             time_units=seeg.time_unit, title=title, offset=0.1, labels=seeg.space_labels,
                             figsize=FiguresConfig.VERY_LARGE_SIZE)

    def plot_simulated_timeseries(self, timeseries, model, seizure_indices, seeg_list=[],
                                  spectral_raster_plot=False, title_prefix="", **kwargs):
        if len(title_prefix) > 0:
            title_prefix = title_prefix + ", " + model._ui_name + ": "
        region_labels = timeseries.space_labels
        state_variables = timeseries.dimension_labels[TimeseriesDimensions.VARIABLES.value]
        source_ts = timeseries.get_source()
        start_plot = int(numpy.round(0.01 * source_ts.data.shape[0]))
        self.plot_raster({'source(t)': source_ts.squeezed[start_plot:, :]},
                         timeseries.time_line.flatten()[start_plot:],
                         time_units=timeseries.time_unit, special_idx=seizure_indices,
                         title=title_prefix + "Simulated source rasterplot", offset=2.0,
                         labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

        if isinstance(model, EpileptorDP2D):
            # We assume that at least x1 and z are available in res
            sv_dict = {'x1(t)': timeseries.x1.squeezed, 'z(t)': timeseries.z.squeezed}

            self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                 special_idx=seizure_indices, title=title_prefix + "Simulated TAVG",
                                 labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

            self.plot_trajectories(sv_dict, special_idx=seizure_indices,
                                   title=title_prefix + 'State space trajectories', labels=region_labels,
                                   figsize=FiguresConfig.LARGE_SIZE)
        else:
            # We assume that at least source and z are available in res
            sv_dict = {'source(t)': source_ts.squeezed, 'z(t)': timeseries.z.squeezed}

            self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                 special_idx=seizure_indices, title=title_prefix + "Simulated source-z",
                                 labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

            if PossibleVariables.X1.value in state_variables and PossibleVariables.Y1.value in state_variables:
                sv_dict = {'x1(t)': timeseries.x1.squeezed, 'y1(t)': timeseries.y1.squeezed}

                self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                     special_idx=seizure_indices, title=title_prefix + "Simulated pop1",
                                     labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)
            if PossibleVariables.X2.value in state_variables and PossibleVariables.Y2.value in state_variables and \
                    PossibleVariables.G.value in state_variables:
                sv_dict = {'x2(t)': timeseries.x2.squeezed, 'y2(t)': timeseries.y2.squeezed,
                           'g(t)': timeseries.g.squeezed}

                self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                     special_idx=seizure_indices, title=title_prefix + "Simulated pop2-g",
                                     labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

            if spectral_raster_plot:
                self.plot_spectral_analysis_raster(timeseries.time_line, source_ts.squeezed,
                                                   time_units=timeseries.time_unit, freq=None,
                                                   special_idx=seizure_indices,
                                                   title=title_prefix + "Spectral Analysis",
                                                   labels=region_labels, figsize=FiguresConfig.LARGE_SIZE, **kwargs)

            if isinstance(model, EpileptorDPrealistic):
                if PossibleVariables.SLOPE_T.value in state_variables and \
                        PossibleVariables.IEXT2_T.value in state_variables:
                    sv_dict = {'1/(1+exp(-10(z-3.03))': 1 / (1 + numpy.exp(-10 * (timeseries.z.squeezed - 3.03))),
                               'slope': timeseries.slope_t.squeezed, 'Iext2': timeseries.Iext2_t.squeezed}
                    title = model._ui_name + ": Simulated controlled parameters"

                    self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                         special_idx=seizure_indices, title=title_prefix + title, labels=region_labels,
                                         figsize=FiguresConfig.VERY_LARGE_SIZE)
                if PossibleVariables.X0_T.value in state_variables and PossibleVariables.IEXT1_T.value in state_variables \
                        and PossibleVariables.K_T.value:
                    sv_dict = {'x0_values': timeseries.x0_t.squeezed, 'Iext1': timeseries.Iext1_t.squeezed,
                               'K': timeseries.K_t.squeezed}

                    self.plot_timeseries(sv_dict, timeseries.time_line, time_units=timeseries.time_unit,
                                         special_idx=seizure_indices,
                                         title=title_prefix + "Simulated parameters",
                                         labels=region_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

        self.plot_simulated_seeg_timeseries(seeg_list, title_prefix=title_prefix)


    def plot_lsa(self, disease_hypothesis, model_configuration, weighted_eigenvector_sum, eigen_vectors_number,
                 region_labels=[], pse_results=None, title="Hypothesis Overview"):

        hyp_dict_list = disease_hypothesis.prepare_for_plot(model_configuration.model_connectivity)
        model_config_dict_list = model_configuration.prepare_for_plot()[:2]

        model_config_dict_list += hyp_dict_list
        plot_dict_list = model_config_dict_list

        if pse_results is not None and isinstance(pse_results, dict):
            fig_name = disease_hypothesis.name + " PSE " + title
            ind_ps = len(plot_dict_list) - 2
            for ii, value in enumerate(["lsa_propagation_strengths", "e_values", "x0_values"]):
                ind = ind_ps - ii
                if ind >= 0:
                    if pse_results.get(value, False).any():
                        plot_dict_list[ind]["data_samples"] = pse_results.get(value)
                        plot_dict_list[ind]["plot_type"] = "vector_violin"

        else:
            fig_name = disease_hypothesis.name + " " + title

        description = ""
        if weighted_eigenvector_sum:
            description = "LSA PS: absolute eigenvalue-weighted sum of "
            if eigen_vectors_number is not None:
                description += "first " + str(eigen_vectors_number) + " "
            description += "eigenvectors has been used"

        return self.plot_in_columns(plot_dict_list, region_labels, width_ratios=[],
                                    left_ax_focus_indices=disease_hypothesis.all_disease_indices,
                                    right_ax_focus_indices=disease_hypothesis.lsa_propagation_indices,
                                    description=description, title=title, figure_name=fig_name)

    def plot_state_space(self, model_config, model="6D", region_labels=[], special_idx=[], zmode="lin", figure_name="",
                         approximations=False, **kwargs):
        add_name = " " + "Epileptor " + model + " z-" + str(zmode)
        figure_name = figure_name + add_name

        region_labels = generate_region_labels(model_config.number_of_regions, region_labels, ". ")
        # n_region_labels = len(region_labels)
        # if n_region_labels == model_config.number_of_regions:
        #     region_labels = numpy.array(["%d. %s" % l for l in zip(range(model_config.number_of_regions), region_labels)])
        # else:
        #     region_labels = numpy.array(["%d" % l for l in range(model_config.number_of_regions)])

        # Fixed parameters for all regions:
        x1eq = model_config.x1eq
        zeq = model_config.zeq
        x0 = a = b = d = yc = slope = Iext1 = Iext2 = s = 0.0
        for p in ["x0", "a", "b", "d", "yc", "slope", "Iext1", "Iext2", "s"]:
            exec (p + " = numpy.mean(model_config." + p + ")")

        fig = pyplot.figure(figure_name, figsize=FiguresConfig.SMALL_SIZE)

        # Lines:
        x1 = numpy.linspace(-2.0, 1.0, 100)
        if isequal_string(model, "2d"):
            y1 = yc
        else:
            y1 = calc_eq_y1(x1, yc, d=d)
        # x1 nullcline:
        zX1 = calc_fx1(x1, z=0, y1=y1, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0, x1_neg=True, model=model,
                       x2=0.0)  # yc + Iext1 - x1 ** 3 - 2.0 * x1 ** 2
        x1null, = pyplot.plot(x1, zX1, 'b-', label='x1 nullcline', linewidth=1)
        ax = pyplot.gca()
        ax.axes.hold(True)
        # z nullcines
        # center point (critical equilibrium point) without approximation:
        # zsq0 = yc + Iext1 - x1sq0 ** 3 - 2.0 * x1sq0 ** 2
        x0e = calc_x0_val_to_model_x0(X0_CR_DEF, yc, Iext1, a=a, b=b, d=d, zmode=zmode)
        x0ne = calc_x0_val_to_model_x0(X0_DEF, yc, Iext1, a=a, b=b, d=d, zmode=zmode)
        zZe = calc_fz(x1, z=0.0, x0=x0e, tau1=1.0, tau0=1.0, zmode=zmode)  # for epileptogenic regions
        zZne = calc_fz(x1, z=0.0, x0=x0ne, tau1=1.0, tau0=1.0, zmode=zmode)  # for non-epileptogenic regions
        zE1null, = pyplot.plot(x1, zZe, 'g-', label='z nullcline at critical point (e_values=1)', linewidth=1)
        zE2null, = pyplot.plot(x1, zZne, 'g--', label='z nullcline for e_values=0', linewidth=1)
        if approximations:
            # The point of the linear approximation (1st order Taylor expansion)
            x1LIN = def_x1lin(X1_DEF, X1EQ_CR_DEF, len(region_labels))
            x1SQ = X1EQ_CR_DEF
            x1lin0 = numpy.mean(x1LIN)
            # The point of the square (parabolic) approximation (2nd order Taylor expansion)
            x1sq0 = numpy.mean(x1SQ)
            # approximations:
            # linear:
            x1lin = numpy.linspace(-5.5 / 3.0, -3.5 / 3, 30)
            # x1 nullcline after linear approximation:
            # yc + Iext1 + 2.0 * x1lin0 ** 3 + 2.0 * x1lin0 ** 2 - \
            # (3.0 * x1lin0 ** 2 + 4.0 * x1lin0) * x1lin  # x1
            zX1lin = calc_fx1_2d_taylor(x1lin, x1lin0, z=0, y1=yc, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0,
                                        x1_neg=None, order=2)  #
            # center point without approximation:
            # zlin0 = yc + Iext1 - x1lin0 ** 3 - 2.0 * x1lin0 ** 2
            # square:
            x1sq = numpy.linspace(-5.0 / 3, -1.0, 30)
            # x1 nullcline after parabolic approximation:
            # + 2.0 * x1sq ** 2 + 16.0 * x1sq / 3.0 + yc + Iext1 + 64.0 / 27.0
            zX1sq = calc_fx1_2d_taylor(x1sq, x1sq0, z=0, y1=yc, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0,
                                       x1_neg=None, order=3, shape=x1sq.shape)
            sq, = pyplot.plot(x1sq, zX1sq, 'm--', label='Parabolic local approximation', linewidth=2)
            lin, = pyplot.plot(x1lin, zX1lin, 'c--', label='Linear local approximation', linewidth=2)
            pyplot.legend(handles=[x1null, zE1null, zE2null, lin, sq])
        else:
            pyplot.legend(handles=[x1null, zE1null, zE2null])

        # Points:
        ii = range(len(region_labels))
        n_special_idx = len(special_idx)
        if n_special_idx > 0:
            ii = numpy.delete(ii, special_idx)
        points = []
        for i in ii:
            point, = pyplot.plot(x1eq[i], zeq[i], '*', mfc='k', mec='k',
                                 ms=10, alpha=0.3, label=str(i) + '.' + region_labels[i])
            points.append(point)
        if n_special_idx > 0:
            for i in special_idx:
                point, = pyplot.plot(x1eq[i], zeq[i], '*', mfc='r', mec='r', ms=10, alpha=0.8,
                                     label=str(i) + '.' + region_labels[i])
                points.append(point)
        # ax.plot(x1lin0, zlin0, '*', mfc='r', mec='r', ms=10)
        # ax.axes.text(x1lin0 - 0.1, zlin0 + 0.2, 'e_values=0.0', fontsize=10, color='r')
        # ax.plot(x1sq0, zsq0, '*', mfc='m', mec='m', ms=10)
        # ax.axes.text(x1sq0, zsq0 - 0.2, 'e_values=1.0', fontsize=10, color='m')

        # Vector field
        tau1 = kwargs.get("tau1", TAU1_DEF)
        tau0 = kwargs.get("tau0", TAU0_DEF)
        X1, Z = numpy.meshgrid(numpy.linspace(-2.0, 1.0, 41), numpy.linspace(0.0, 6.0, 31), indexing='ij')
        if isequal_string(model, "2d"):
            y1 = yc
            x2 = 0.0
        else:
            y1 = calc_eq_y1(X1, yc, d=d)
            x2 = 0.0  # as a simplification for faster computation without important consequences
            # x2 = calc_eq_x2(Iext2, y2eq=None, zeq=X1, x1eq=Z, s=s)[0]
        fx1 = calc_fx1(X1, Z, y1=y1, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=tau1, x1_neg=None, model=model,
                       x2=x2)
        fz = calc_fz(X1, Z, x0=x0, tau1=tau1, tau0=tau0, zmode=zmode)
        C = numpy.abs(fx1) + numpy.abs(fz)
        pyplot.quiver(X1, Z, fx1, fz, C, edgecolor='k', alpha=.5, linewidth=.5)
        pyplot.contour(X1, Z, fx1, 0, colors='b', linestyles="dashed")

        ax.set_title("Epileptor states pace at the x1-z phase plane of the" + add_name)
        ax.axes.autoscale(tight=True)
        ax.axes.set_ylim([0.0, 6.0])
        ax.axes.set_xlabel('x1')
        ax.axes.set_ylabel('z')

        if self.config.figures.MOUSE_HOOVER:
            self.HighlightingDataCursor(points[0], formatter='{label}'.format, bbox=dict(fc='white'),
                                        arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))

        if len(fig.get_label()) == 0:
            fig.set_label(figure_name)
        else:
            figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")

        self._save_figure(None, figure_name)
        self._check_show()

    def _prepare_distribution_axes(self, distribution, loc=0.0, scale=1.0, x=numpy.array([]), ax=None, linestyle="-",
                                   lgnd=False):
        if len(x) < 1:
            x = linspace_broadcast(distribution._scipy_method("ppf", distribution.loc, distribution.scale, 0.01),
                                   distribution._scipy_method("ppf", distribution.loc, distribution.scale,0.99), 100)
        if x is not None:
            if x.ndim == 1:
                x = x[:, numpy.newaxis]
            pdf = distribution._scipy_method("pdf", loc, scale, x)
            if ax is None:
                _, ax = pyplot.subplots(1, 1)
            for ip, (xx, pp) in enumerate(zip(x.T, pdf.T)):
                ax.plot(xx.T, pp.T, linestyle=linestyle, linewidth=1, label=str(ip), alpha=0.5)
            if lgnd:
                pyplot.legend()
            return ax
        else:
            # TODO: is this message correct??
            raise_value_error("Distribution parameters do not broadcast!")

    def plot_distribution(self, distribution, loc=0.0, scale=1.0, x=numpy.array([]), ax=None, linestyle="-", lgnd=True,
                          figure_name=""):
        ax = self._prepare_distribution_axes(loc, scale, x, ax, linestyle, lgnd)
        ax.set_title(distribution.type + " distribution")
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return ax, pyplot.gcf()

    def _prepare_parameter_axes(self, parameter, x=numpy.array([]), ax=None, lgnd=False):
        if ax is None:
            _, ax = pyplot.subplots(1, 1)
        x, pdf = parameter.scipy_method("pdf", x)
        if x.ndim == 1:
            x = x[:, numpy.newaxis]
        x, pdf = parameter.scipy_method("pdf", x)
        if ax is None:
            _, ax = pyplot.subplots(1, 1)
        for ip, (xx, pp) in enumerate(zip(x.T, pdf.T)):
            ax.plot(xx.T, pp.T, linestyle="-", linewidth=1, label=str(ip), alpha=0.5)
        if lgnd:
            pyplot.legend()
        ax.set_title(parameter.name + ": " + parameter.type + " distribution")
        return ax

    def plot_probabilistic_parameter(self, parameter, x=numpy.array([]), ax=None, lgnd=True, figure_name=""):
        ax = self._prepare_parameter_axes(parameter, x, ax, lgnd)
        if len(figure_name) < 1:
            figure_name = "parameter_" + parameter.name
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return ax, pyplot.gcf()

    def plot_HMC(self, samples, skip_samples=0, title='HMC NUTS trace', figure_name=None,
                 figsize=FiguresConfig.LARGE_SIZE):
        samples = ensure_list(samples)
        if len(samples) > 1:
            samples = list_of_dicts_to_dicts_of_ndarrays(samples)
        else:
            samples = samples[0]
        nuts = extract_dict_stringkeys(samples, '__', modefun="find")
        self.plots(nuts, shape=(2, 4), transpose=True, skip=skip_samples, xlabels={}, xscales={},
                   yscales={"stepsize__": "log"}, title=title, figure_name=figure_name, figsize=figsize)

    def _params_stats_subtitles(self, params, stats):
        subtitles = list(params)
        if isinstance(stats, dict):
            for ip, param in enumerate(params):
                subtitles[ip] = subtitles[ip] + ": "
                for skey, sval in stats.iteritems():
                    subtitles[ip] = subtitles[ip] + skey + "=" + str(sval[param]) + ", "
                subtitles[ip] = subtitles[ip][:-2]
        return subtitles

    def _params_stats_labels(self, param, stats, labels):
        subtitles = list(labels)
        if isinstance(stats, dict):
            n_params = len(stats.values()[0][param])
            if len(subtitles) == 1 and n_params > 1:
                subtitles = subtitles * n_params
            elif len(subtitles) == 0:
                subtitles = [""] * n_params
            for ip in range(n_params):
                if len(subtitles[ip]) > 0:
                    subtitles[ip] = subtitles[ip] + ": "
                for skey, sval in stats.iteritems():
                    subtitles[ip] = subtitles[ip] + skey + "=" + str(sval[param][ip]) + ", "
                    subtitles[ip] = subtitles[ip][:-2]
        return subtitles

    def parameters_pair_plots(self, samples, params=["tau1",  "K", "sigma", "epsilon", "scale", "offset"], stats=None,
                              priors={}, truth={}, skip_samples=0, title='Parameters samples', figure_name=None,
                              figsize=FiguresConfig.VERY_LARGE_SIZE):
        subtitles = list(self._params_stats_subtitles(params, stats))
        samples = ensure_list(samples)
        if len(samples) > 1:
            samples = list_of_dicts_to_dicts_of_ndarrays(samples)
        else:
            samples = samples[0]
        # samples = sort_dict(extract_dict_stringkeys(samples, params, modefun="equal"))
        diagonal_plots = {}
        # for param_key in samples.keys():
        for param_key in params:
            diagonal_plots.update({param_key: [priors.get(param_key, ()), truth.get(param_key, ())]})

        return self.pair_plots(samples, params, diagonal_plots, True, skip_samples,
                                    title, subtitles, figure_name, figsize)

    def region_parameters_violin_plots(self, samples, values=None, lines=None, stats=None,
                                       params=["x0", "x1init", "zinit"], skip_samples=0, per_chain=False, labels=[],
                                       seizure_indices=None, figure_name="Regions parameters samples",
                                       figsize=FiguresConfig.VERY_LARGE_SIZE):
        if isinstance(values, dict):
            vals_fun = lambda param: values.get(param, numpy.array([]))
        else:
            vals_fun = lambda param: []
        if isinstance(lines, dict):
            lines_fun = lambda param: lines.get(param, numpy.array([]))
        else:
            lines_fun = lambda param: []
        samples = ensure_list(samples)
        n_chains = len(samples)
        if not per_chain and len(samples) > 1:
            samples = ensure_list(list_of_dicts_to_dicts_of_ndarrays(samples))
            plot_samples = lambda s: numpy.concatenate(numpy.split(s[:, skip_samples:].T, n_chains, axis=2),
                                                       axis=1).squeeze().T
            plot_figure_name = lambda ichain: figure_name
        else:
            plot_samples = lambda s: s[skip_samples:]
            plot_figure_name = lambda ichain: figure_name + ": chain " + str(ichain + 1)
        labels = generate_region_labels(samples[0][params[0]].shape[-1], labels)
        params_labels = {}
        for ip, p in enumerate(params):
            if ip == 0:
                params_labels[p] = self._params_stats_labels(p, stats, labels)
            else:
                params_labels[p] = self._params_stats_labels(p, stats, "")
        n_params = len(params)
        if n_params > 9:
            warning("Number of subplots in column wise vector-violin-plots cannot be > 9 and it is "
                              + str(n_params) + "!")
        subplot_ind = 100 + n_params * 10
        for ichain, chain_sample in enumerate(samples):
            pyplot.figure(plot_figure_name(ichain), figsize=figsize)
            for ip, param in enumerate(params):
                self.plot_vector_violin(plot_samples(chain_sample[param]), vals_fun(param),
                                        lines_fun(param), params_labels[param],
                                        subplot_ind + ip + 1, param, colormap="YlOrRd", show_y_labels=True,
                                        indices_red=seizure_indices, sharey=None)
            self._save_figure(pyplot.gcf(), None)
            self._check_show()

    def plot_fit_scalar_params(self, samples, stats, probabilistic_model=None,
                               pair_plot_params=["tau1",  "K", "sigma", "epsilon", "scale", "offset"], skip_samples=0,
                               title_prefix=""):
        # plot scalar parameters in pair plots
        if len(title_prefix) > 0:
            title_prefix = title_prefix + ": "
        priors = {}
        truth = {}
        if probabilistic_model is not None:
            title = title_prefix + probabilistic_model.name + " parameters samples"
            for p in pair_plot_params:
                priors.update({p: probabilistic_model.get_prior_pdf(p)})
                truth.update({p: numpy.nanmean(probabilistic_model.get_truth(p))})
        else:
            title = title_prefix + "Parameters samples"

        self.parameters_pair_plots(samples, pair_plot_params, stats, priors, truth, skip_samples, title=title)

    def plot_fit_region_params(self, samples, stats=None, probabilistic_model=None,
                               region_violin_params=["x0", "x1init", "zinit"], seizure_indices=[], region_labels=[],
                               regions_mode="all", per_chain_plotting=False, skip_samples=0, title_prefix=""):
        if len(title_prefix) > 0:
            title_prefix = title_prefix + " "
        # We assume in this function that regions_inds run for all regions for the statistical model,
        # and either among all or only among active regions for samples, ests and stats, depending on regions_mode
        samples = ensure_list(samples)
        priors = {}
        truth = {}
        if probabilistic_model is not None:
            title_pair_plot = title_prefix + probabilistic_model.name + " global coupling vs x0 pair plot"
            title_violin_plot = title_prefix + probabilistic_model.name + " regions parameters samples"
            if regions_mode=="active":
                regions_inds = probabilistic_model.active_regions
            else:
                regions_inds = range(probabilistic_model.number_of_regions)
            I = numpy.ones((probabilistic_model.number_of_regions, 1))
            for param in region_violin_params:
                pdf = ensure_list(probabilistic_model.get_prior_pdf(param))
                for ip, p in enumerate(pdf):
                    pdf[ip] = ((p.T * I)[regions_inds])
                priors.update({param: (pdf[0].squeeze(), pdf[1].squeeze())})
                truth.update({param: ((probabilistic_model.get_truth(param) * I[:, 0])[regions_inds]).squeeze()})
        else:
            title_pair_plot = title_prefix + probabilistic_model.name + "Global coupling vs x0 pair plot"
            title_violin_plot = title_prefix + probabilistic_model.name + "Regions parameters samples"
            regions_inds = range(samples[0]["x0"].shape[2])
        # plot region-wise parameters
        self.region_parameters_violin_plots(samples, truth, priors, stats, region_violin_params, skip_samples,
                                            per_chain=per_chain_plotting, labels=region_labels,
                                            seizure_indices=seizure_indices, figure_name=title_violin_plot)
        if not(per_chain_plotting) and "x0" in region_violin_params and samples[0]["x0"].shape[1] < 10:
            x0_K_pair_plot_params = []
            x0_K_pair_plot_samples = [{} for _ in range(len(samples))]
            if samples[0].get("K", None) is not None:
                # plot K-x0 parameters in pair plots
                x0_K_pair_plot_params = ["K"]
                x0_K_pair_plot_samples = [{"K": s["K"]} for s in samples]
                if probabilistic_model is not None:
                    priors.update({"K": probabilistic_model.get_prior_pdf("K")})
                    truth.update({"K": probabilistic_model.get_truth("K")})
            for inode, label in enumerate(region_labels):
                temp_name = "x0[" + label + "]"
                x0_K_pair_plot_params.append(temp_name)
                for ichain, s in enumerate(samples):
                    x0_K_pair_plot_samples[ichain].update({temp_name: s["x0"][:, inode]})
                    if probabilistic_model is not None:
                        priors.update({temp_name: (priors["x0"][0][inode], priors["x0"][1][inode])})
                        truth.update({temp_name: truth["x0"][inode]})
            self.parameters_pair_plots(x0_K_pair_plot_samples, x0_K_pair_plot_params, None, priors, truth, skip_samples,
                                       title=title_pair_plot)

    def plot_fit_timeseries(self, target_data, samples, ests, stats=None, probabilistic_model=None,
                            seizure_indices=[], skip_samples=0, trajectories_plot=False, title_prefix=""):
        if len(title_prefix) > 0:
            title_prefix = title_prefix + ": "
        samples = ensure_list(samples)
        region_labels = samples[0]["x1"].space_labels
        if probabilistic_model is not None:
            sig_prior_str = " sig_prior = " + str(probabilistic_model.get_prior("sigma")[0])
        else:
            sig_prior_str = ""
        stats_region_labels = region_labels
        if stats is not None:
            stats_string = {"fit_target_data": "\n", "x1": "\n", "z": "\n", "MC": ""}
            if isinstance(stats, dict):
                for skey, sval in stats.iteritems():
                    for p_str in ["fit_target_data", "x1", "z"]:
                        stats_string[p_str] \
                            = stats_string[p_str] + skey + "_mean=" + str(numpy.mean(sval[p_str])) + ", "
                    stats_region_labels = [stats_region_labels[ip] + ", " +
                                           skey + "_" + "x1" + "_mean=" + str(sval["x1"][:, ip].mean()) + ", " +
                                           skey + "_z_mean=" + str(sval["z"][:, ip].mean())
                                           for ip in range(len(region_labels))]
                for p_str in ["fit_target_data", "x1", "z"]:
                    stats_string[p_str] = stats_string[p_str][:-2]
        else:
            stats_string = dict(zip(["target_data", "x1", "z"], 3*[""]))
        observation_dict = OrderedDict({'observation time series': target_data.squeezed})
        time = target_data.time_line
        for id_est, (est, sample) in enumerate(zip(ensure_list(ests), samples)):
            name = title_prefix + probabilistic_model.name + "_chain" + str(id_est + 1)
            observation_dict.update({"fit chain " + str(id_est + 1):
                                         sample["fit_target_data"].squeezed[:, :, skip_samples:]})
            self.plot_raster(sort_dict({"x1": sample["x1"].squeezed[:, :, skip_samples:],
                                        'z': sample["z"].squeezed[:, :, skip_samples:]}),
                             time, special_idx=seizure_indices, time_units=target_data.time_unit,
                             title=name + ": Hidden states fit rasterplot",
                             subtitles=['hidden state ' + "x1" + stats_string["x1"],
                                        'hidden state z' + stats_string["z"]], offset=1.0,
                             labels=region_labels,
                             figsize=FiguresConfig.VERY_LARGE_SIZE)
            dWt = {}
            subtitles = []
            if sample.get("dX1t", None):
                dWt.update({"dX1t": sample["dX1t"].squeezed[:, :, skip_samples:]})
                subtitles.append("dX1t")
            if sample.get("dZt", None):
                dWt.update({"dZt": sample["dZt"].squeezed[:, :, skip_samples:]})
                subtitles.append("dZt")
            if len(dWt) > 0:
                subtitles[-1] += "\ndynamic noise" + sig_prior_str + ", sig_post = " + str(est["sigma"])
                self.plot_raster(sort_dict(dWt), time[:-1], time_units=target_data.time_unit,
                                 special_idx=seizure_indices, title=name + ": Hidden states random walk rasterplot",
                                 subtitles=subtitles, offset=1.0, labels=region_labels,
                                 figsize=FiguresConfig.VERY_LARGE_SIZE)
            if trajectories_plot:
                title = name + ' Fit hidden state space trajectories'
                self.plot_trajectories({"x1": sample["x1"].squeezed[:, :, skip_samples:],
                                        'z': sample['z'].squeezed[:, :, skip_samples:]},
                                       special_idx=seizure_indices, title=title, labels=stats_region_labels,
                                       figsize=FiguresConfig.SUPER_LARGE_SIZE)
        self.plot_raster(observation_dict, time, special_idx=[], time_units=target_data.time_unit,
                         title=title_prefix + probabilistic_model.name + "Observation target vs fit time series: "
                                + stats_string["fit_target_data"],
                         figure_name=title_prefix + probabilistic_model.name + "ObservationTarget_VS_FitTimeSeries",
                         offset=1.0, labels=target_data.space_labels, figsize=FiguresConfig.VERY_LARGE_SIZE)

    def plot_fit_connectivity(self, ests, samples, stats=None, probabilistic_model=None, region_labels=[], title_prefix=""):
        # plot connectivity
        if len(title_prefix) > 0:
            title_prefix = title_prefix + "_"
        if probabilistic_model is not None:
            name0 = title_prefix + probabilistic_model.name + "_"
            MC_prior = probabilistic_model.get_prior("MC")
            MC_subplot = 122
        else:
            name0 = title_prefix
            MC_prior = False
            MC_subplot = 111
        for id_est, (est, sample) in enumerate(zip(ensure_list(ests), ensure_list(samples))):
            conn_figure_name = name0 + "chain" + str(id_est + 1) + ": Model Connectivity"
            pyplot.figure(conn_figure_name, FiguresConfig.VERY_LARGE_SIZE)
            # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
            if MC_prior:
                self.plot_regions2regions(MC_prior, region_labels, 121,
                                          "Prior Model Connectivity")
            MC_title = "Posterior Model  Connectivity"
            if isinstance(stats, dict):
                MC_title = MC_title + ": "
                for skey, sval in stats.iteritems():
                    MC_title = MC_title + skey + "_mean=" + str(sval["MC"].mean()) + ", "
                MC_title = MC_title[:-2]
            self.plot_regions2regions(est["MC"], region_labels, MC_subplot, MC_title)
            self._save_figure(pyplot.gcf(), conn_figure_name)
            self._check_show()

    def plot_fit_results(self, ests, samples, model_data, target_data, probabilistic_model=None, stats=None,
                         pair_plot_params=["tau1", "K", "sigma", "epsilon", "scale", "offset"],
                         region_violin_params=["x0", "x1init", "zinit"],
                         regions_labels=[], regions_mode="active", n_regions=1,
                         trajectories_plot=True, connectivity_plot=False, skip_samples=0, title_prefix=""):
        if probabilistic_model is not None:
            active_regions = probabilistic_model.active_regions
        else:
            active_regions = model_data.get("active_regions", range(n_regions))
            regions_labels = generate_region_labels(n_regions, regions_labels, ". ", False)
        if isequal_string(regions_mode, "all"):
            seizure_indices = active_regions
        else:
            region_inds = active_regions
            seizure_indices = None
            regions_labels = regions_labels[region_inds]

        self.plot_fit_scalar_params(samples, stats, probabilistic_model, pair_plot_params, skip_samples, title_prefix)

        self.plot_fit_region_params(samples, stats, probabilistic_model, region_violin_params, seizure_indices,
                                    regions_labels, regions_mode, False, skip_samples, title_prefix)

        self.plot_fit_region_params(samples, stats, probabilistic_model, region_violin_params, seizure_indices,
                                    regions_labels, regions_mode, True, skip_samples, title_prefix)

        self.plot_fit_timeseries(target_data, samples, ests, stats, probabilistic_model,
                                 seizure_indices, skip_samples, trajectories_plot, title_prefix)

        if connectivity_plot:
            self.plot_fit_connectivity(ests, stats, probabilistic_model, regions_labels, title_prefix)
