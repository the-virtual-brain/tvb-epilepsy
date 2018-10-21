# coding=utf-8

from tvb_fit.base.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)
from matplotlib import pyplot, gridspec
from matplotlib.colors import Normalize

import numpy

from tvb_fit.base.utils.log_error_utils import warning, raise_value_error
from tvb_fit.base.utils.data_structures_utils import ensure_list, isequal_string, \
                                                          generate_region_labels, ensure_string
from tvb_fit.base.computations.analyzers_utils import time_spectral_analysis
from tvb_fit.plot.base_plotter import BasePlotter


class TimeseriesPlotter(BasePlotter):

    linestyle = "-"
    linewidth = 1
    marker = None
    markersize = 2
    markerfacecolor = None

    def __init__(self, config=None):
        super(TimeseriesPlotter, self).__init__(config)
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

    @property
    def line_format(self):
        return {"linestyle": self.linestyle, "linewidth": self.linewidth,
                "marker": self.marker, "markersize": self.markersize, "markerfacecolor": self.markerfacecolor}

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

        def plot_ts(x, iTS, colors, labels):
            x, time, ivar = x
            try:
                return pyplot.plot(time, x[:, iTS], color=colors[iTS], label=labels[iTS], **self.line_format)
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(time, x[:, iTS], color=colors[iTS], label=str(iTS), **self.line_format)

        def plot_ts_raster(x, iTS, colors, labels, offset):
            x, time, ivar = x
            try:
                return pyplot.plot(time, -x[:, iTS] + (offset * iTS + x[:, iTS].mean()), color=colors[iTS],
                                   label=labels[iTS], **self.line_format)
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(time, -x[:, iTS] + offset * iTS, color=colors[iTS],
                                   label=str(iTS),  **self.line_format)

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
            plot_lines = lambda x, iTS, colors, labels: \
                 plot_ts_raster(x, iTS, colors,  labels, offset)
        else:
            plot_lines = lambda x, iTS, colors, labels: \
                        plot_ts(x, iTS, colors, labels)
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

        def plot_traj_2D(x, iTS, colors, labels):
            x, y = x
            try:
                return pyplot.plot(x[:, iTS], y[:, iTS], color=colors[iTS], label=labels[iTS], **self.line_format)
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(x[:, iTS], y[:, iTS], color=colors[iTS], label=str(iTS), **self.line_format)

        def plot_traj_3D(x, iTS, colors, labels):
            x, y, z = x
            try:
                return pyplot.plot(x[:, iTS], y[:, iTS], z[:, iTS], color=colors[iTS],
                                   label=labels[iTS], **self.line_format)
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(x[:, iTS], y[:, iTS], z[:, iTS], color=colors[iTS],
                                   label=str(iTS), **self.line_format)

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
            plot_lines = lambda x, iTS, colors, labels: \
                   plot_traj_2D(x, iTS, colors, labels)
            projection = None
        elif n_dims == 3:
            plot_lines = lambda x, iTS, colors, labels: \
                   plot_traj_3D(x, iTS, colors, labels)
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

    # TODO: refactor to not have the plot commands here
    def plot_timeseries(self, data_dict, time=None, mode="ts", subplots=None, special_idx=[], subtitles=[], labels=[],
                        offset=1.0, time_units="ms", title='Time series', figure_name=None,
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
        if special_idx is None:
            special_idx = []
        n_special_idx = len(special_idx)
        if len(subtitles) == 0:
            subtitles = vars
        labels = generate_region_labels(nTS, labels)
        if isequal_string(mode, "traj"):
            data_fun, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
            subtitle, subtitle_col, axlabels, axlimits = \
                self._trajectories_plot(n_vars, nTS, nSamples, subplots)
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
        alphas = numpy.maximum(numpy.array([def_alpha] * nTS) * alpha_ratio, 0.1)
        alphas[special_idx] = numpy.maximum(alpha_ratio, 0.1)
        if (n_vars / (n_cols*n_rows) == 1) or not isequal_string(mode, "raster"): # (isequal_string(mode, "raster") and n_special_idx == 0):
            cmap = matplotlib.cm.get_cmap('jet')
            colors = numpy.array([cmap(0.8 * iTS / nTS) for iTS in range(nTS)])
            colors[special_idx] = \
                numpy.array([cmap(1.0 - 0.2 * iTS / nTS) for iTS in range(n_special_idx)]).reshape((n_special_idx, 4))
        else:
            colors = numpy.zeros((nTS, 4))
            colors[special_idx] = \
                numpy.array([numpy.array([1.0, 0, 0, 1.0]) for _ in range(n_special_idx)]).reshape((n_special_idx, 4))
        colors[:, 3] = alphas
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
                lines += ensure_list(plot_lines(data_fun(data, time, icol), iTS, colors, labels))
            if isequal_string(mode, "raster"):  # set yticks as labels if this is a raster plot
                axYticks(labels, nTS, icol)
                if n_special_idx > 0:
                    yticklabels = pyplot.gca().yaxis.get_ticklabels()
                    for special_id in special_idx:
                        yticklabels[special_id].set_color(colors[special_id, :3].tolist() + [1])
                    pyplot.gca().yaxis.set_ticklabels(yticklabels)
                pyplot.gca().invert_yaxis()

        if self.config.figures.MOUSE_HOOVER:
            for line in lines:
                self.HighlightingDataCursor(line, formatter='{label}'.format, bbox=dict(fc='white'),
                                            arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))

        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return pyplot.gcf(), axes, lines

    def plot_raster(self, data_dict, time, time_units="ms", special_idx=[], title='Raster plot', subtitles=[],
                    labels=[], offset=1.0, figure_name=None, figsize=FiguresConfig.VERY_LARGE_SIZE):
        return self.plot_timeseries(data_dict, time, "raster", None, special_idx, subtitles, labels, offset, time_units,
                                    title, figure_name, figsize)

    def plot_trajectories(self, data_dict, subtitles=None, special_idx=[], labels=[], title='State space trajectories',
                          figure_name=None, figsize=FiguresConfig.LARGE_SIZE):
        return self.plot_timeseries(data_dict, [], "traj", subtitles, special_idx, labels=labels, title=title,
                                     figure_name=figure_name, figsize=figsize)

    # TODO: refactor to not have the plot commands here
    def plot_spectral_analysis_raster(self, time, data, time_units="ms", freq=None, spectral_options={},
                                      special_idx=[], labels=[], title='Spectral Analysis', figure_name=None,
                                      figsize=FiguresConfig.VERY_LARGE_SIZE):
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
        log_norm = spectral_options.get("log_norm", False)
        mode = spectral_options.get("mode", "psd")
        psd_label = mode
        if log_norm:
            psd_label = "log" + psd_label
        stf, time, freq, psd = time_spectral_analysis(data, fs,
                                                      freq=freq,
                                                      mode=mode,
                                                      nfft=spectral_options.get("nfft"),
                                                      window=spectral_options.get("window", 'hanning'),
                                                      nperseg=spectral_options.get("nperseg", int(numpy.round(fs / 4))),
                                                      detrend=spectral_options.get("detrend", 'constant'),
                                                      noverlap=spectral_options.get("noverlap"),
                                                      f_low=spectral_options.get("f_low", 10.0),
                                                      log_scale=spectral_options.get("log_scale", False))
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
                # TODO: find and correct bug here
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
